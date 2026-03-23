"""Generate dataset README.md files using Gemini.

This module pre-analyzes the files in a dataset directory to extract hard facts
(column counts, port numbers, file extensions, value ranges), then asks Gemini
to produce the README schema.  After generation, the schema is validated with
the same parser the loader uses; if validation fails the error is sent back to
Gemini for correction (up to 2 retries).
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# The system prompt teaches the LLM the exact schema format.
_SYSTEM_PROMPT = """\
You are a dataset documentation assistant for a surrogate model training tool.

Your job: given pre-analyzed facts about a dataset directory AND sample file
content, produce a README.md file with a fenced ```json``` schema block.

## CRITICAL RULES — the parser is strict about these

- `ground_truth_parts` MUST use EXACTLY these values: "re", "im", "mag", "db", "angle_deg"
  Do NOT write "real", "imaginary", "magnitude", "phase", etc.
- `columns` MUST have EXACTLY the number of entries matching the actual column count provided.
  Do NOT add or remove columns. Use the pre-analyzed column count as ground truth.
  If the log.txt has a header comment with column names (starting with #), use THOSE EXACT names.
- `sample_id_column` MUST be one of the names in `columns`.
- `feature_columns` MUST be a subset of `columns`. Exclude sample IDs, thread IDs, and batch IDs.
  DO include constant columns (like na, nb, outa, outb, outbound) — the loader auto-drops them.
- Only include fields documented below. Do NOT invent fields like "frequency_unit".
- For `ground_truth_parameters`: use the LOWER-TRIANGULAR S-parameters for an N-port device.
  For 2-port: S11, S12, S22 (3 params).
  For 4-port: S11, S12, S13, S14, S22, S23, S24, S33, S34, S44 (10 params).
  For 6-port: all Sij where j >= i (21 params).
  Do NOT list only the first row (S11,S21,S31,S41). List ALL unique lower-triangular entries.
  Note: S12=S21, S13=S31, etc. by reciprocity. Use the form Sij where i <= j.

## Supported formats

### 1. Touchstone (S-parameter .sNp files + log.txt)
Directory has: `log.txt` (one Python list per line, optionally with a # header comment) and `SPData/` folder with `.sNp` files.

GOLD-STANDARD EXAMPLE — follow this structure exactly for touchstone datasets:
```json
{
  "dataset_name": "XFMR_2508_1x1_DiffXY",
  "input_feature": {
    "columns": [
      "rax", "ray", "rbx", "rby", "na", "nb",
      "wida", "widb", "gapa", "gapb",
      "opena", "openb", "outa", "outb",
      "exta", "extb", "dist", "ratio", "outbound",
      "index", "batch"
    ],
    "feature_columns": [
      "rax", "ray", "rbx", "rby", "na", "nb",
      "wida", "widb",
      "opena", "openb", "outa", "outb",
      "exta", "extb", "dist", "ratio", "outbound"
    ],
    "sample_id_column": "index"
  },
  "ground_truth": {
    "format": "touchstone",
    "file_extension": ".s4p",
    "ground_truth_parameters": [
      "S11", "S12", "S13", "S14",
      "S22", "S23", "S24",
      "S33", "S34",
      "S44"
    ],
    "ground_truth_parts": ["re", "im"],
    "drop_first_frequency": false
  }
}
```

Rules for touchstone:
- `columns` count MUST match the pre-analyzed column count exactly.
- If log.txt has a # header comment with column names, use those exact names for `columns`.
- Otherwise, name columns by inspecting the data values (geometry params, IDs, etc.)
- `feature_columns`: exclude sample IDs (index), thread/batch IDs. KEEP all geometry parameters
  even if constant — the loader auto-drops constants. This ensures the schema stays valid if
  a future dataset has non-constant values in those columns.
- `sample_id_column`: the column whose values match filenames in SPData/ (0.s4p → value 0).
  Usually the second-to-last column.
- `ground_truth_parameters`: ALL unique lower-triangular S-parameters for the N-port device.
  For 4-port (.s4p) → 10 params: S11,S12,S13,S14,S22,S23,S24,S33,S34,S44.
  For 2-port (.s2p) → 3 params: S11,S12,S22.
- `ground_truth_parts`: MUST be from ["re", "im", "mag", "db", "angle_deg"]. Typically ["re", "im"].
- `drop_first_frequency`: true ONLY if the first frequency is exactly 0.0 Hz (DC).
  If pre-analyzed first_frequency is > 0, set to false.

### 2. Cadence CSV (block-structured simulation exports)
Each CSV has parameter blocks separated by markers like "CS = ...", with swept data.

Example schema:
```json
{
  "dataset_name": "CTLE_gain",
  "input_feature": {
    "columns": ["CS_fF", "LD_pH", "M", "RD", "RS", "MN"],
    "feature_columns": ["CS_fF", "LD_pH", "M", "RD", "RS", "MN"],
    "source": "embedded",
    "parameter_keys": {
      "CS": ["CS_fF", 1e15],
      "LD": ["LD_pH", 1e12],
      "M": ["M", 1],
      "RD": ["RD", 1],
      "RS": ["RS", 1],
      "MN": ["MN", 1]
    }
  },
  "ground_truth": {
    "format": "cadence_csv",
    "channel_files": {
      "gain": "CTLE_gain_1000sample.csv"
    },
    "channel_units": {
      "gain": "dB"
    }
  }
}
```

Rules for cadence_csv:
- `source`: always "embedded"
- `parameter_keys`: map CSV key → [stored_name, scale]. Common scales: F→fF=1e15, H→pH=1e12.
  For keys whose values have no SI suffix and are plain numbers (like M, RD, RS, MN), use scale=1.
- `channel_files`: map channel name → CSV filename.
- `channel_units`: physical unit of output extracted from the CSV header line (e.g. "dB", "deg", "V/sqrt(Hz)")
- `channel_transforms`: set to "log10" ONLY if data spans many orders of magnitude (>1000x range). Otherwise omit entirely.

## Output format

Write a complete README.md with:
1. A title (# Dataset Name)
2. A brief description
3. The fenced ```json``` schema block
4. Any relevant notes about the dataset

IMPORTANT: The JSON block MUST be valid JSON inside triple-backtick json fences.
"""


def _pre_analyze(directory: Path) -> dict:
    """Extract hard facts from the dataset directory that the LLM must not guess."""
    facts: dict = {"directory": str(directory), "files": [], "subdirs": []}

    for f in sorted(directory.iterdir()):
        if f.is_dir():
            sub_files = sorted(f.iterdir())
            facts["subdirs"].append({
                "name": f.name,
                "file_count": len(sub_files),
                "sample_extensions": [sf.suffix for sf in sub_files[:3]],
            })
        else:
            facts["files"].append(f.name)

    # Analyze log.txt if present.
    log_path = directory / "log.txt"
    if log_path.exists():
        header_comment = None
        first_data_line = None
        data_line_count = 0
        with log_path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    header_comment = line
                    continue
                if first_data_line is None:
                    first_data_line = line
                data_line_count += 1
        try:
            row = ast.literal_eval(first_data_line)
            facts["log_txt"] = {
                "total_lines": data_line_count,
                "columns_per_row": len(row),
                "first_row_values": row,
                "constant_columns": _find_constant_columns(log_path, len(row)),
            }
            if header_comment:
                facts["log_txt"]["header_comment"] = header_comment
        except Exception:
            facts["log_txt"] = {"total_lines": data_line_count, "first_line": first_data_line}

    # Analyze SPData if present.
    sp_dir = directory / "SPData"
    if sp_dir.is_dir():
        sp_files = sorted(sp_dir.iterdir())
        if sp_files:
            ext = sp_files[0].suffix.lower()
            port_match = re.fullmatch(r"\.s(\d+)p", ext)
            ports = int(port_match.group(1)) if port_match else None
            # Read first touchstone file header.
            header_info = _read_touchstone_header(sp_files[0])
            # Compute lower-triangular S-parameters for N-port.
            lower_tri_params = []
            if ports:
                for i in range(1, ports + 1):
                    for j in range(i, ports + 1):
                        lower_tri_params.append(f"S{i}{j}")
            facts["spdata"] = {
                "file_count": len(sp_files),
                "extension": ext,
                "port_count": ports,
                "lower_triangular_sparams": lower_tri_params,
                "sample_ids": [int(sf.stem) for sf in sp_files[:5]],
                **header_info,
            }

    # Analyze CSV files if present.
    csv_files = [f for f in directory.iterdir() if f.suffix.lower() == ".csv"]
    if csv_files:
        facts["csv_files"] = []
        for csv_path in sorted(csv_files)[:5]:
            facts["csv_files"].append(_analyze_csv(csv_path))

    return facts


def _find_constant_columns(log_path: Path, ncols: int, max_rows: int = 100) -> list[int]:
    """Find column indices that have constant values across samples."""
    import numpy as np
    rows = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= max_rows:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(ast.literal_eval(line))
            except Exception:
                pass
    if len(rows) < 2:
        return []
    arr = np.array(rows, dtype=float)
    return [int(i) for i in range(ncols) if arr[:, i].std() < 1e-8]


def _read_touchstone_header(path: Path) -> dict:
    """Read metadata from the first few lines of a touchstone file."""
    info: dict = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            lower = line.lower()
            if line.startswith("#"):
                parts = line[1:].split()
                if len(parts) >= 3:
                    info["frequency_unit"] = parts[0]
                    info["data_format"] = parts[2]  # RI, MA, DB
            elif lower.startswith("[number of ports]"):
                info["number_of_ports"] = int(line.split("]", 1)[1].strip())
            elif lower.startswith("[number of frequencies]"):
                info["number_of_frequencies"] = int(line.split("]", 1)[1].strip())
            elif lower.startswith("[matrix format]"):
                info["matrix_format"] = line.split("]", 1)[1].strip()
            elif lower.startswith("[end]"):
                break
            elif lower.startswith("[network data]"):
                # Next non-empty line is the first data row.
                continue
            elif not line.startswith("["):
                # First data line — check if first freq is 0
                try:
                    first_freq = float(line.split()[0])
                    info["first_frequency"] = first_freq
                except (ValueError, IndexError):
                    pass
                break
    return info


def _analyze_csv(path: Path) -> dict:
    """Extract structure info from a Cadence-style CSV file."""
    _SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
           "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

    def _try_si(s: str) -> float | None:
        s = s.strip()
        if not s:
            return None
        try:
            if s[-1] in _SI:
                return float(s[:-1]) * _SI[s[-1]]
            return float(s)
        except (ValueError, IndexError):
            return None

    info: dict = {"filename": path.name}
    header_line = ""
    param_keys: list[dict] = []
    data_values: list[float] = []
    in_first_block = False

    with path.open(encoding="utf-8") as fh:
        for i, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            # Capture the very first long line as header.
            if i < 3 and len(line) > 60 and not header_line:
                header_line = line[:120]
                continue
            if len(line) > 60:
                continue
            if "=" in line:
                key = line.split("=", 1)[0].strip()
                val = line.split("=", 1)[1].strip()
                if not in_first_block and not param_keys:
                    in_first_block = True
                if in_first_block and not any(pk["key"] == key for pk in param_keys):
                    param_keys.append({"key": key, "example_value": val})
            else:
                parts = line.split()
                if len(parts) == 2 and in_first_block:
                    v = _try_si(parts[1])
                    if v is not None:
                        data_values.append(v)
            # Stop after first block is complete (next param block starts).
            if in_first_block and data_values and "=" in line:
                break

    info["header"] = header_line
    info["parameter_keys"] = param_keys
    info["data_rows_in_first_block"] = len(data_values)
    if data_values:
        import numpy as np
        arr = np.array(data_values)
        positive = arr[arr > 0]
        info["value_min"] = float(arr.min())
        info["value_max"] = float(arr.max())
        if len(positive) > 1:
            info["dynamic_range"] = float(positive.max() / positive.min())
            info["recommend_log10"] = float(positive.max() / positive.min()) > 1000
    return info


def _sample_files(directory: Path, max_lines: int = 20, max_files: int = 8) -> str:
    """Read the first few lines of files for the LLM context."""
    parts: list[str] = []
    files = sorted(directory.iterdir())
    sampled = 0

    for f in files:
        if sampled >= max_files:
            break
        if f.is_dir():
            sub_files = sorted(f.iterdir())
            if sub_files:
                parts.append(f"=== {f.name}/{sub_files[0].name} (first {max_lines} lines) ===")
                try:
                    with sub_files[0].open(encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh):
                            if i >= max_lines:
                                break
                            parts.append(line.rstrip())
                except Exception:
                    parts.append("(binary or unreadable)")
                parts.append(f"... ({len(sub_files)} files in {f.name}/)")
                sampled += 1
        elif f.suffix.lower() in (".csv", ".txt", ".log", ".s2p", ".s3p", ".s4p", ".s6p", ".s8p"):
            parts.append(f"=== {f.name} (first {max_lines} lines) ===")
            try:
                with f.open(encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if i >= max_lines:
                            break
                        parts.append(line.rstrip())
            except Exception:
                parts.append("(binary or unreadable)")
            sampled += 1

    return "\n".join(parts)


def _validate_readme(readme_text: str) -> str | None:
    """Try to parse the README schema. Return None on success, error message on failure."""
    import tempfile
    from .dataset_schema import parse_dataset_readme

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(readme_text)
            tmp_path = f.name
        parse_dataset_readme(tmp_path)
        return None
    except Exception as e:
        return str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def generate_readme(
    directory: str | Path,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-flash-lite",
    max_retries: int = 2,
    progress_callback=None,
) -> str:
    """Generate a README.md for the given dataset directory using Gemini.

    The function pre-analyzes the directory to extract hard facts, generates
    the README, then validates it with the schema parser.  On validation
    failure, it sends the error back to Gemini for correction.
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package is required for README generation. "
            "Install it with: pip install google-genai"
        )

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key required. Set GOOGLE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    if progress_callback:
        progress_callback({"event": "analyzing", "message": "Pre-analyzing dataset files..."})

    # Step 1: Pre-analyze to extract hard facts.
    facts = _pre_analyze(directory)
    file_preview = _sample_files(directory)

    user_message = (
        f"Generate a README.md for this dataset.\n\n"
        f"## Pre-analyzed facts (DO NOT contradict these):\n"
        f"```\n{_format_facts(facts)}\n```\n\n"
        f"## Sample file content:\n{file_preview}"
    )

    if progress_callback:
        progress_callback({"event": "generating", "message": "Sending to Gemini for analysis..."})

    client = genai.Client(api_key=key)
    config = genai.types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.1,
    )

    # Step 2: Generate and validate, with retries on failure.
    contents = [user_message]
    readme_text = ""

    for attempt in range(1 + max_retries):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        readme_text = response.text

        # Step 3: Validate.
        error = _validate_readme(readme_text)
        if error is None:
            if progress_callback:
                progress_callback({"event": "done", "message": "README generated and validated successfully."})
            return readme_text

        if attempt < max_retries:
            if progress_callback:
                progress_callback({
                    "event": "retrying",
                    "message": f"Validation failed, asking Gemini to fix (attempt {attempt + 2})...",
                })
            # Send the error back for correction.
            contents = [
                user_message,
                readme_text,
                (
                    f"The README you generated failed schema validation with this error:\n"
                    f"```\n{error}\n```\n"
                    f"Please fix the JSON schema block and regenerate the complete README.md."
                ),
            ]

    # Return the last attempt even if validation failed — the user can fix it in the editor.
    if progress_callback:
        progress_callback({
            "event": "done",
            "message": f"WARNING: Generated README has validation issues: {error}",
        })
    return readme_text


def _format_facts(facts: dict) -> str:
    """Format pre-analyzed facts as a readable string for the LLM prompt."""
    lines = []
    if "log_txt" in facts:
        log = facts["log_txt"]
        lines.append(f"log.txt: {log.get('total_lines', '?')} data lines, {log.get('columns_per_row', '?')} columns per row")
        if "header_comment" in log:
            lines.append(f"  COLUMN NAMES (from header): {log['header_comment']}")
            lines.append(f"  USE THESE EXACT NAMES for 'columns' in the schema.")
        if "first_row_values" in log:
            lines.append(f"  First row values: {log['first_row_values']}")
        if "constant_columns" in log and log["constant_columns"]:
            lines.append(f"  Constant columns (indices): {log['constant_columns']}")

    if "spdata" in facts:
        sp = facts["spdata"]
        lines.append(f"SPData/: {sp['file_count']} files, extension={sp['extension']}, ports={sp.get('port_count', '?')}")
        lines.append(f"  Sample IDs: {sp.get('sample_ids', [])}")
        if sp.get("lower_triangular_sparams"):
            lines.append(f"  USE THESE EXACT ground_truth_parameters: {sp['lower_triangular_sparams']}")
        if "number_of_frequencies" in sp:
            lines.append(f"  Frequencies: {sp['number_of_frequencies']}, first_freq={sp.get('first_frequency', '?')}")
            first_f = sp.get("first_frequency")
            if first_f is not None:
                lines.append(f"  drop_first_frequency: {'true' if first_f == 0.0 else 'false'}")
        if "matrix_format" in sp:
            lines.append(f"  Matrix format: {sp['matrix_format']}, data format: {sp.get('data_format', '?')}")

    if "csv_files" in facts:
        for csv in facts["csv_files"]:
            lines.append(f"CSV: {csv['filename']}")
            if csv.get("header"):
                lines.append(f"  Header: {csv['header'][:100]}")
            if csv.get("parameter_keys"):
                keys_str = ", ".join(f"{pk['key']}={pk['example_value']}" for pk in csv["parameter_keys"])
                lines.append(f"  Parameters: {keys_str}")
            lines.append(f"  Data points per sample: {csv.get('data_rows_in_first_block', '?')}")
            if "value_min" in csv:
                lines.append(f"  Value range: {csv['value_min']:.4e} to {csv['value_max']:.4e}")
            if csv.get("dynamic_range"):
                lines.append(f"  Dynamic range: {csv['dynamic_range']:.0f}x {'(RECOMMEND log10 transform)' if csv.get('recommend_log10') else ''}")

    lines.append(f"All files/dirs: {facts.get('files', [])} + subdirs: {[s['name'] for s in facts.get('subdirs', [])]}")
    return "\n".join(lines)


def save_readme(directory: str | Path, content: str) -> Path:
    """Write the README.md to the dataset directory and return its path."""
    path = Path(directory) / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Multi-turn Chat Session for interactive README generation
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT = """\
You are a helpful AI assistant embedded in a surrogate-model training GUI.
Your primary role is to help users understand their datasets and generate
correct README.md schema files so the training pipeline can load the data.

You have access to pre-analyzed facts about the user's dataset directory.
When the user asks you to generate or fix a README, use the exact same JSON
schema format described below.

""" + _SYSTEM_PROMPT.split("## Output format")[0] + """\
## Important behaviour rules

- When the user asks you to generate a README, output the COMPLETE README.md
  content (title, description, fenced json block, notes) inside a single
  markdown code block fenced with ```readme ... ```.
- When the user asks questions about their data, answer concisely based on
  the pre-analyzed facts and file previews.
- If the user asks you to fix or change specific fields, output the full
  updated README.md (not just the changed part) inside ```readme ... ```.
- Keep answers concise and focused on the dataset / README task.
"""


class ChatSession:
    """Multi-turn conversation session with Gemini for dataset assistance."""

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash-lite",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.history: list[dict] = []  # [{"role": "user"/"model", "text": str}]
        self._facts: dict | None = None
        self._file_preview: str | None = None

    def set_dataset(self, dataset_dir: str | Path) -> str:
        """Set or change the dataset directory. Returns a summary of what was found."""
        self.dataset_dir = Path(dataset_dir)
        self._facts = _pre_analyze(self.dataset_dir)
        self._file_preview = _sample_files(self.dataset_dir)
        summary = _format_facts(self._facts)
        return f"Dataset loaded: {self.dataset_dir.name}\n{summary}"

    def send(self, user_message: str) -> str:
        """Send a message and get a response. Blocking call."""
        try:
            from google import genai
        except ImportError:
            return "Error: google-genai package not installed. Run: pip install google-genai"

        if not self.api_key:
            return "Error: No Gemini API key configured. Set GOOGLE_API_KEY environment variable or enter it in Settings."

        # Auto-analyze dataset on first message if not done yet.
        if self._facts is None and self.dataset_dir and self.dataset_dir.is_dir():
            self.set_dataset(self.dataset_dir)

        # Build the context-enriched first user message.
        context_block = ""
        if self._facts:
            context_block = (
                f"\n\n[Dataset context — pre-analyzed facts]\n"
                f"```\n{_format_facts(self._facts)}\n```\n\n"
                f"[Sample file content]\n{self._file_preview}\n\n"
            )

        # Build contents for the API call (full history).
        contents = []
        for msg in self.history:
            contents.append(genai.types.Content(
                role=msg["role"],
                parts=[genai.types.Part(text=msg["text"])],
            ))

        # Add the new user message (with context on first message only).
        enriched_message = user_message
        if not self.history and context_block:
            enriched_message = user_message + context_block
        elif context_block and not any("Dataset context" in m["text"] for m in self.history):
            enriched_message = user_message + context_block

        contents.append(genai.types.Content(
            role="user",
            parts=[genai.types.Part(text=enriched_message)],
        ))

        try:
            client = genai.Client(api_key=self.api_key)
            config = genai.types.GenerateContentConfig(
                system_instruction=_CHAT_SYSTEM_PROMPT,
                temperature=0.2,
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            reply = response.text
        except Exception as e:
            reply = f"Error: {e}"

        # Store in history (store enriched version so context is in history).
        self.history.append({"role": "user", "text": enriched_message})
        self.history.append({"role": "model", "text": reply})

        return reply

    def extract_readme(self, text: str) -> str | None:
        """Extract README content from a ```readme ... ``` or ```json ... ``` fenced block in the response."""
        import re
        # Try ```readme first, then ```markdown, then look for the json schema block.
        for pattern in [
            r"```readme\s*\n(.*?)```",
            r"```markdown\s*\n(.*?)```",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        # Fallback: if the response contains a ```json block with "dataset_name",
        # it's likely the full README.
        if '"dataset_name"' in text and "```json" in text:
            # Return the full text as-is (it's already a README).
            return text.strip()
        return None

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()
