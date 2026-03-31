"""LLM-powered role classifier for scanned dataset files.

Given a ``ScanResult`` from ``dataset_scanner``, this module asks an LLM
(Gemini) to classify each file/group into one of a fixed set of roles:

- **input_parameters**: tabular file containing design/geometric parameters
  (one row per sample).
- **ground_truth**: simulation output data (S-parameters, gain curves, etc.)
  that the surrogate model will learn to predict.
- **layout**: pixel / image representations of physical designs.
- **metadata**: port info, auxiliary lookup tables, or other supporting data.
- **ignore**: files not needed for training (documentation, scripts, etc.).

The LLM sees *only* the metadata extracted by the scanner (column names,
shapes, value ranges) — never raw data.  Its job is semantic classification,
not schema generation.  The result is a list of ``RoleAssignment`` objects
that the wizard UI presents for user confirmation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .dataset_scanner import FileInfo, ScanResult, format_scan_for_llm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# The fixed set of roles the classifier can assign.
ROLES = ("input_parameters", "ground_truth", "layout", "metadata", "ignore")

ROLE_DESCRIPTIONS = {
    "input_parameters": "Design / geometric parameters (one row per sample)",
    "ground_truth": "Simulation output the model learns to predict",
    "layout": "Pixel or image representation of physical designs",
    "metadata": "Supporting data (port info, auxiliary tables)",
    "ignore": "Not needed for training",
}


@dataclass
class RoleAssignment:
    """One file/group with a suggested role and optional notes."""

    file_index: int
    file_info: FileInfo
    role: str
    confidence: str = "high"   # "high", "medium", "low"
    reason: str = ""
    # Columns identified as feature columns (for input_parameters role).
    feature_columns: list[str] = field(default_factory=list)
    # Columns identified as the sample ID column.
    id_column: str = ""


@dataclass
class ClassificationResult:
    """Full classification output for a scanned dataset."""

    assignments: list[RoleAssignment]
    # Free-form dataset summary from the LLM.
    dataset_summary: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = """\
You are a dataset role classifier for a surrogate-model training tool.

You will receive metadata about files found in a dataset directory.  Each file
entry has an index like [0], [1], etc.  Your job: classify every entry into
exactly one role.

## Roles

- **input_parameters**: A tabular file (CSV, Excel, txt, etc.) where each row
  is one sample and columns are design/geometric parameters.  Typically has
  a small number of columns (< 30) and many rows matching the sample count.
  Look for column names like dimensions, widths, turns, spacing, ratios, etc.

- **ground_truth**: The simulation output that the model will predict.
  Could be: per-sample CSV/touchstone files with frequency-swept S-parameters,
  Cadence-style block CSVs, pre-packed pickle/numpy arrays with shape
  [num_samples, num_frequencies, num_channels], etc.

- **layout**: Pixel or image data representing the physical design.
  Could be: PNG images, numpy arrays with shape [num_samples, H, W] or
  [H, W] per sample, GDS files, pixel CSVs, etc.

- **metadata**: Supporting info — port coordinates, config files, notebooks,
  auxiliary lookup tables.  Useful but not directly model input or output.

- **ignore**: Documentation, scripts, temporary files, logs not needed for
  training.

## Rules

1. Classify EVERY file entry.  Do not skip any.
2. Use the column names, shapes, and value ranges to decide — they are facts.
3. If a tabular CSV has columns that look like geometric parameters AND a
   separate column that looks like a sample ID/name, it is input_parameters.
4. If files in a directory have columns like freq, s11_real, s11_imag, or if
   touchstone files exist, those are ground_truth.
5. Numpy/pickle arrays with 3D shape [N, freq, channels] are ground_truth.
   Arrays with shape [N, H, W] are layout.
6. For input_parameters entries, also specify which columns are features
   (exclude ID/name columns) and which column is the sample ID.
7. Be concise.  Respond ONLY with valid JSON, no markdown fences.

## Response format

Respond with a JSON object:
{
  "dataset_summary": "One-line description of what this dataset appears to contain",
  "assignments": [
    {
      "index": 0,
      "role": "input_parameters",
      "confidence": "high",
      "reason": "Tabular CSV with geometric parameter columns",
      "feature_columns": ["col1", "col2"],
      "id_column": "sample_name"
    },
    {
      "index": 1,
      "role": "ground_truth",
      "confidence": "high",
      "reason": "Per-sample CSVs with S-parameter columns"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_dataset(
    scan: ScanResult,
    *,
    api_key: str | None = None,
    model_name: str = "gemini-3-flash-preview",
) -> ClassificationResult:
    """Use an LLM to classify every file/group in *scan* into a role.

    Requires a Gemini API key.  If unavailable, returns all files as
    ``unclassified`` so the user can assign roles manually in the wizard.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not key:
        return _unclassified(scan, "No API key — assign roles manually")

    try:
        return _classify_with_llm(scan, key, model_name)
    except Exception as exc:
        return _unclassified(scan, f"LLM error: {exc} — assign roles manually")


def _classify_with_llm(
    scan: ScanResult,
    api_key: str,
    model_name: str,
) -> ClassificationResult:
    """Call Gemini to classify the scanned files."""
    from google import genai

    facts_text = format_scan_for_llm(scan)
    user_message = f"Classify every file entry below:\n\n{facts_text}"

    client = genai.Client(api_key=api_key)
    config = genai.types.GenerateContentConfig(
        system_instruction=_CLASSIFIER_PROMPT,
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[user_message],
        config=config,
    )

    return _parse_llm_response(response.text, scan)


def _parse_llm_response(text: str, scan: ScanResult) -> ClassificationResult:
    """Parse the JSON response from the LLM into a ClassificationResult."""
    # Strip markdown fences if present.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    payload = json.loads(text)
    assignments: list[RoleAssignment] = []

    for entry in payload.get("assignments", []):
        idx = int(entry["index"])
        if idx < 0 or idx >= len(scan.files):
            continue
        role = entry.get("role", "ignore")
        if role not in ROLES:
            role = "ignore"
        assignments.append(RoleAssignment(
            file_index=idx,
            file_info=scan.files[idx],
            role=role,
            confidence=entry.get("confidence", "medium"),
            reason=entry.get("reason", ""),
            feature_columns=entry.get("feature_columns", []),
            id_column=entry.get("id_column", ""),
        ))

    # Ensure every file has an assignment.
    assigned_indices = {a.file_index for a in assignments}
    for i, fi in enumerate(scan.files):
        if i not in assigned_indices:
            assignments.append(RoleAssignment(
                file_index=i,
                file_info=fi,
                role="ignore",
                confidence="low",
                reason="Not classified by LLM",
            ))

    assignments.sort(key=lambda a: a.file_index)
    return ClassificationResult(
        assignments=assignments,
        dataset_summary=payload.get("dataset_summary", ""),
    )


# ---------------------------------------------------------------------------
# Fallback when no LLM is available
# ---------------------------------------------------------------------------

def _unclassified(scan: ScanResult, reason: str) -> ClassificationResult:
    """Return all files as unclassified so the user assigns roles manually."""
    return ClassificationResult(
        assignments=[
            RoleAssignment(
                file_index=i,
                file_info=fi,
                role="ignore",
                confidence="low",
                reason=reason,
            )
            for i, fi in enumerate(scan.files)
        ],
        dataset_summary=reason,
    )
