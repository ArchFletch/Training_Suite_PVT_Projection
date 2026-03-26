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
content, produce a README.md that accurately describes the dataset structure,
geometric/input parameters, ground-truth data, and how to load it.

## CRITICAL RULES

1. **Describe what you SEE, not what you assume.** The pre-analyzed facts tell
   you the exact column names, file counts, directory structure, value ranges,
   and array shapes. Use ONLY those facts. Never copy column names or parameters
   from a different dataset.
2. **Column names must come from the data.** If a tabular CSV header says
   `NUM_TURNS,LINE_WIDTH,SPACING,WIDTH,TOTAL_LENGTH,R_OD,sample_name`, those
   are the columns — not `rax, ray, rbx, rby` or any other names.
3. **feature_columns** = the subset of columns that are geometric/design
   parameters (inputs to the model). Exclude sample IDs, batch/thread IDs,
   and file-name columns.
4. **ground_truth_parameters**: for S-parameter data, list the LOWER-TRIANGULAR
   unique entries.  For 2-port: S11, S12, S22.  For 4-port: S11, S12, S13,
   S14, S22, S23, S24, S33, S34, S44.  Use form Sij where i <= j.
5. **ground_truth_parts** MUST use EXACTLY these values where applicable:
   "re", "im", "mag", "db", "angle_deg".
6. **drop_first_frequency**: true ONLY if first frequency is exactly 0.0 Hz (DC).
7. If the dataset has multiple templates/variants (e.g. different geometries
   stored in separate subdirectories), describe ALL of them. Note which
   parameters vary vs. which are constant per template.

## Detecting the dataset format

Examine the pre-analyzed facts to determine what kind of dataset this is.
Common patterns include (but are NOT limited to):

### inline (single CSV with all data)
- A single CSV file contains ALL samples. Each sample spans multiple rows
  (one per frequency point). Rows sharing the same feature-column values
  form one sample.
- `source`: "inline". Set `input_feature.file_path` to the CSV.
- MUST set `frequency_column` to the name of the swept-axis column
  (e.g. "freq (Hz)", "Freq_GHz", "VDIFF").
- If the swept axis is NOT frequency, set `sweep_label` to a display label
  (e.g. "VDIFF (mV)"). Defaults to "Frequency (GHz)" if omitted.
- Feature columns are constant within a sample; GT columns vary per sweep point.
- GT column names are auto-detected from common conventions (e.g.
  `S11_re`, `s11_real`, `S11` for dB, `S11_phase` for angle_deg).
- Special case: Cadence block-structured CSVs use `channel_files` dict and
  `input_feature.source: "embedded"` with `parameter_keys`.

### per_sample (one GT file per sample)
- Separate GT file per sample (CSV or Touchstone `.sNp`) in a `data_dir/`.
- Input features in a separate tabular CSV with `sample_id_column` for linking.
- `source`: "per_sample". Set `file_extension` (e.g. ".csv", ".s2p").
- For CSV files, optionally set `frequency_column`.

### array (pre-packed numpy)
- Pre-packed numpy array in a pickle file, shape (samples, frequencies, channels).
- `source`: "array". Set `data_dir` to the pickle file path.
- Include `frequency_start_hz` and `frequency_stop_hz` (in Hz).

### Other / mixed formats
- Some datasets have layout images, GDS files, pixel arrays, port info, etc.
- Describe ALL data modalities found in the directory.

## Output format

Write a complete README.md with:
1. **Title** (# Dataset Name)
2. **Description** — what this dataset contains, what device/circuit it models.
3. **Templates/variants** — if multiple subdirectory groups exist, list them
   with sample counts.
4. **Input features** — table of geometric parameter names, descriptions,
   units, and value ranges (from the pre-analyzed facts).
5. **Ground truth** — what output data is stored, format, frequency range,
   which S-parameters or channels, column names from the actual CSV headers.
6. **Directory structure** — a tree showing the layout.
7. **Data loading** — code snippet showing how to load the data.
8. **Schema block** — a fenced ```json``` block describing the dataset schema
   for the training pipeline. Use fields appropriate to the detected format.
   The schema MUST include these path fields so the loader knows where to find files:
   - `input_feature.file_path`: relative path from dataset root to the input-feature
     file (e.g. `"log.txt"`, `"tabular/data.csv"`). Omit only for embedded formats.
   - `ground_truth.data_dir`: relative path from dataset root to the directory
     containing ground-truth files (e.g. `"SPData"`, `"csv/oct_l"`).
9. **Notes** — any important caveats (constant columns, DC frequency, etc.)

IMPORTANT: The JSON block MUST be valid JSON inside triple-backtick json fences.
Do NOT copy examples from other datasets. Build the schema from the actual facts.
"""


def _pre_analyze(directory: Path) -> dict:
    """Extract hard facts from the dataset directory that the LLM must not guess.

    This function is format-agnostic: it inspects the directory tree and reports
    whatever it finds (tabular CSVs with headers, per-sample data files, pickle
    arrays, touchstone files, Cadence-style CSVs, GDS/image files, etc.) so
    that the LLM can describe the dataset accurately regardless of its layout.
    """
    facts: dict = {"directory": str(directory), "files": [], "subdirs": []}

    for f in sorted(directory.iterdir()):
        if f.is_dir():
            facts["subdirs"].append(_scan_subdir(f))
        else:
            facts["files"].append(f.name)

    # --- Analyze log.txt if present (touchstone workflow) ---
    log_path = directory / "log.txt"
    if log_path.exists():
        facts["log_txt"] = _analyze_log_txt(log_path)

    # --- Analyze SPData if present (touchstone workflow) ---
    sp_dir = directory / "SPData"
    if sp_dir.is_dir():
        facts["spdata"] = _analyze_spdata(sp_dir)

    # --- Analyze top-level CSV files ---
    csv_files = [f for f in directory.iterdir() if f.suffix.lower() == ".csv"]
    if csv_files:
        facts["csv_files"] = []
        for csv_path in sorted(csv_files)[:5]:
            facts["csv_files"].append(_analyze_csv_auto(csv_path))

    # --- Analyze tabular CSV files in subdirectories ---
    # Look for directories named "tabular" or CSV files with header rows in subdirs.
    tabular_csvs = _find_tabular_csvs(directory)
    if tabular_csvs:
        facts["tabular_csvs"] = tabular_csvs

    # --- Analyze pickle files ---
    pkl_files = sorted(directory.rglob("*.pkl"))
    if pkl_files:
        facts["pkl_files"] = _analyze_pkl_files(pkl_files[:10])

    # --- Analyze per-sample data directories ---
    # Detect subdirectories that contain many similarly-named data files (CSVs, sNp, etc.)
    per_sample_dirs = _find_per_sample_dirs(directory)
    if per_sample_dirs:
        facts["per_sample_dirs"] = per_sample_dirs

    # --- Detect image / layout directories ---
    image_dirs = _find_image_dirs(directory)
    if image_dirs:
        facts["image_dirs"] = image_dirs

    # --- Detect port info ---
    port_info = _find_port_info(directory)
    if port_info:
        facts["port_info"] = port_info

    return facts


def _scan_subdir(d: Path, max_depth: int = 2) -> dict:
    """Recursively scan a subdirectory up to max_depth levels."""
    entries = sorted(d.iterdir())
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir()]
    info: dict = {
        "name": d.name,
        "file_count": len(files),
        "dir_count": len(dirs),
    }
    if files:
        info["sample_extensions"] = list({f.suffix for f in files})
    if dirs and max_depth > 1:
        info["subdirs"] = [_scan_subdir(sd, max_depth - 1) for sd in dirs[:8]]
    elif dirs:
        info["subdir_names"] = [sd.name for sd in dirs]
    return info


def _analyze_log_txt(log_path: Path) -> dict:
    """Analyze a log.txt file (touchstone workflow)."""
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
        result: dict = {
            "total_lines": data_line_count,
            "columns_per_row": len(row),
            "first_row_values": row,
            "constant_columns": _find_constant_columns(log_path, len(row)),
        }
        if header_comment:
            result["header_comment"] = header_comment
        return result
    except Exception:
        return {"total_lines": data_line_count, "first_line": first_data_line}


def _analyze_spdata(sp_dir: Path) -> dict:
    """Analyze an SPData directory with touchstone files."""
    sp_files = sorted(sp_dir.iterdir())
    if not sp_files:
        return {"file_count": 0}
    ext = sp_files[0].suffix.lower()
    port_match = re.fullmatch(r"\.s(\d+)p", ext)
    ports = int(port_match.group(1)) if port_match else None
    header_info = _read_touchstone_header(sp_files[0])
    lower_tri_params = []
    if ports:
        for i in range(1, ports + 1):
            for j in range(i, ports + 1):
                lower_tri_params.append(f"S{i}{j}")
    return {
        "file_count": len(sp_files),
        "extension": ext,
        "port_count": ports,
        "lower_triangular_sparams": lower_tri_params,
        "sample_ids": [int(sf.stem) for sf in sp_files[:5]],
        **header_info,
    }


def _analyze_csv_auto(csv_path: Path) -> dict:
    """Analyze a CSV file, auto-detecting whether it's tabular or Cadence-style."""
    # First check if it has a standard CSV header row.
    try:
        with csv_path.open(encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            fh.readline()  # skip second line
    except Exception:
        return {"filename": csv_path.name, "error": "unreadable"}

    # If the first line looks like a header (no numbers, comma-separated words)
    # treat it as a tabular CSV.
    if first_line and "=" not in first_line and not first_line.startswith("!"):
        parts = first_line.split(",")
        is_header = all(not _is_numeric(p.strip()) for p in parts) and len(parts) >= 2
        if is_header:
            return _analyze_tabular_csv(csv_path, first_line)

    # Fall back to Cadence-style block analysis.
    return _analyze_csv(csv_path)


def _is_numeric(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _analyze_tabular_csv(csv_path: Path, header_line: str) -> dict:
    """Analyze a CSV file with a standard header row."""
    columns = [c.strip() for c in header_line.split(",")]
    row_count = 0
    first_row_values: list[str] = []
    with csv_path.open(encoding="utf-8") as fh:
        fh.readline()  # skip header
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            if i == 0:
                first_row_values = [v.strip() for v in line.strip().split(",")]
            row_count += 1
    info: dict = {
        "filename": csv_path.name,
        "type": "tabular",
        "columns": columns,
        "column_count": len(columns),
        "row_count": row_count,
    }
    if first_row_values:
        info["first_row_values"] = first_row_values
    return info


def _find_tabular_csvs(directory: Path) -> list[dict]:
    """Find CSV files with header rows in the directory tree (e.g. tabular/ subdir)."""
    results = []
    # Check common subdir names for tabular data.
    for subdir_name in ("tabular", "params", "features", "metadata"):
        subdir = directory / subdir_name
        if subdir.is_dir():
            for csv_path in sorted(subdir.glob("*.csv"))[:8]:
                results.append(_analyze_tabular_csv_full(csv_path))
    return results


def _analyze_tabular_csv_full(csv_path: Path) -> dict:
    """Deep analysis of a tabular CSV: columns, value ranges, constant columns."""
    import numpy as np
    info: dict = {"filename": csv_path.name, "path": str(csv_path.parent.name) + "/" + csv_path.name}
    try:
        with csv_path.open(encoding="utf-8") as fh:
            header = fh.readline().strip()
        columns = [c.strip() for c in header.split(",")]
        info["columns"] = columns
        info["column_count"] = len(columns)

        # Read the data to find value ranges and constant columns.
        rows = []
        id_column_idx = None
        with csv_path.open(encoding="utf-8") as fh:
            fh.readline()  # skip header
            for line in fh:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                row_numeric = []
                for j, val in enumerate(parts):
                    val = val.strip()
                    if _is_numeric(val):
                        row_numeric.append(float(val))
                    else:
                        row_numeric.append(float("nan"))
                        if id_column_idx is None:
                            id_column_idx = j
                rows.append(row_numeric)

        info["row_count"] = len(rows)
        if rows:
            arr = np.array(rows)
            # Find numeric vs non-numeric columns.
            numeric_cols = []
            constant_cols = []
            value_ranges = {}
            for j, col_name in enumerate(columns):
                if j < arr.shape[1]:
                    col_data = arr[:, j]
                    non_nan = col_data[~np.isnan(col_data)]
                    if len(non_nan) > 0:
                        numeric_cols.append(col_name)
                        value_ranges[col_name] = {
                            "min": float(non_nan.min()),
                            "max": float(non_nan.max()),
                        }
                        if len(non_nan) > 1 and non_nan.std() < 1e-8:
                            constant_cols.append(col_name)
            info["numeric_columns"] = numeric_cols
            info["value_ranges"] = value_ranges
            if constant_cols:
                info["constant_columns"] = constant_cols
            if id_column_idx is not None and id_column_idx < len(columns):
                info["id_column"] = columns[id_column_idx]
    except Exception as e:
        info["error"] = str(e)
    return info


def _analyze_pkl_files(pkl_paths: list[Path]) -> list[dict]:
    """Analyze pickle files to report their types and shapes."""
    results = []
    for pkl_path in pkl_paths:
        info: dict = {"filename": pkl_path.name, "path": str(pkl_path.relative_to(pkl_path.parent.parent))}
        try:
            import pickle
            import numpy as np
            with pkl_path.open("rb") as fh:
                data = pickle.load(fh)
            info["type"] = type(data).__name__
            if hasattr(data, "shape"):
                info["shape"] = list(data.shape)
            elif isinstance(data, (list, tuple)):
                info["length"] = len(data)
            if isinstance(data, np.ndarray):
                info["dtype"] = str(data.dtype)
        except Exception as e:
            info["error"] = str(e)
        results.append(info)
    return results


def _find_per_sample_dirs(directory: Path) -> list[dict]:
    """Find directories containing many per-sample data files."""
    results = []
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        # Check for nested template dirs (e.g. csv/oct_l/, csv/rec_r/).
        nested_dirs = [d for d in sorted(subdir.iterdir()) if d.is_dir()]
        if nested_dirs:
            for nested in nested_dirs[:6]:
                info = _analyze_data_dir(nested, prefix=f"{subdir.name}/{nested.name}")
                if info:
                    results.append(info)
        else:
            info = _analyze_data_dir(subdir, prefix=subdir.name)
            if info:
                results.append(info)
    return results


def _analyze_data_dir(d: Path, prefix: str) -> dict | None:
    """Analyze a directory of per-sample data files. Returns None if not a data dir."""
    files = sorted(f for f in d.iterdir() if f.is_file())
    if len(files) < 5:
        return None

    extensions = {}
    for f in files:
        ext = f.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    dominant_ext = max(extensions, key=extensions.get) if extensions else ""

    info: dict = {
        "path": prefix,
        "file_count": len(files),
        "dominant_extension": dominant_ext,
    }

    # Read the first file to get column info (for CSVs) or header info (for sNp).
    first_file = files[0]
    if dominant_ext == ".csv":
        try:
            with first_file.open(encoding="utf-8") as fh:
                header = fh.readline().strip()
                fh.readline()  # skip first data line
                line_count = 2 + sum(1 for _ in fh)
            columns = [c.strip() for c in header.split(",")]
            if all(not _is_numeric(c) for c in columns):
                info["csv_columns"] = columns
                info["csv_column_count"] = len(columns)
                info["rows_per_file"] = line_count - 1  # minus header
        except Exception:
            pass
    elif re.fullmatch(r"\.s\d+p", dominant_ext):
        info["touchstone_header"] = _read_touchstone_header(first_file)

    return info


def _find_image_dirs(directory: Path) -> list[dict]:
    """Find directories containing image files (PNG, SVG, numpy, etc.)."""
    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".npy", ".npz", ".bmp", ".tif", ".tiff"}
    results = []
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        # Check nested dirs (e.g. PNG/oct_l/).
        nested_dirs = [d for d in sorted(subdir.iterdir()) if d.is_dir()]
        if nested_dirs:
            for nested in nested_dirs[:6]:
                files = [f for f in nested.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
                if files:
                    results.append({
                        "path": f"{subdir.name}/{nested.name}",
                        "file_count": len(files),
                        "extension": files[0].suffix.lower(),
                    })
        else:
            files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
            if files:
                results.append({
                    "path": subdir.name,
                    "file_count": len(files),
                    "extension": files[0].suffix.lower(),
                })
    return results


def _find_port_info(directory: Path) -> list[dict]:
    """Find port info files (CSVs with port coordinates)."""
    results = []
    port_dir = directory / "port_info"
    if not port_dir.is_dir():
        return results
    for subdir in sorted(port_dir.iterdir()):
        if subdir.is_dir():
            csv_files = sorted(subdir.glob("*.csv"))
            if csv_files:
                try:
                    with csv_files[0].open(encoding="utf-8") as fh:
                        header = fh.readline().strip()
                        first_row = fh.readline().strip()
                    results.append({
                        "path": f"port_info/{subdir.name}",
                        "file_count": len(csv_files),
                        "columns": header,
                        "example": first_row,
                    })
                except Exception:
                    pass
    return results


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


_TEXT_EXTENSIONS = {".csv", ".txt", ".log", ".s2p", ".s3p", ".s4p", ".s6p", ".s8p", ".tsv"}


def _sample_files(directory: Path, max_lines: int = 20, max_files: int = 12) -> str:
    """Read the first few lines of files for the LLM context.

    Handles nested directory structures (up to 2 levels deep) so that datasets
    organized as e.g. ``csv/oct_l/sample_0.csv`` or ``tabular/data.csv`` are
    properly sampled.
    """
    parts: list[str] = []
    sampled = 0

    def _read_file(path: Path, label: str) -> bool:
        nonlocal sampled
        if sampled >= max_files:
            return False
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            return False
        parts.append(f"=== {label} (first {max_lines} lines) ===")
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= max_lines:
                        break
                    parts.append(line.rstrip())
        except Exception:
            parts.append("(binary or unreadable)")
        sampled += 1
        return True

    for entry in sorted(directory.iterdir()):
        if sampled >= max_files:
            break
        if entry.is_file():
            _read_file(entry, entry.name)
        elif entry.is_dir():
            # Check for nested subdirectories (e.g. csv/oct_l/, tabular/).
            sub_entries = sorted(entry.iterdir())
            sub_dirs = [e for e in sub_entries if e.is_dir()]
            sub_files = [e for e in sub_entries if e.is_file()]

            if sub_dirs:
                # Nested structure: sample one file from each nested subdir.
                for sd in sub_dirs[:4]:
                    if sampled >= max_files:
                        break
                    nested_files = sorted(sd.iterdir())
                    readable = [f for f in nested_files if f.is_file() and f.suffix.lower() in _TEXT_EXTENSIONS]
                    if readable:
                        label = f"{entry.name}/{sd.name}/{readable[0].name}"
                        _read_file(readable[0], label)
                        parts.append(f"... ({len(nested_files)} files in {entry.name}/{sd.name}/)")
            elif sub_files:
                # Flat subdir: sample the first readable file.
                readable = [f for f in sub_files if f.suffix.lower() in _TEXT_EXTENSIONS]
                if readable:
                    label = f"{entry.name}/{readable[0].name}"
                    _read_file(readable[0], label)
                    parts.append(f"... ({len(sub_files)} files in {entry.name}/)")

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

    # --- Directory overview ---
    lines.append(f"Directory: {facts.get('directory', '?')}")
    lines.append(f"Top-level files: {facts.get('files', [])}")
    if facts.get("subdirs"):
        lines.append("Top-level subdirectories:")
        for sd in facts["subdirs"]:
            detail = f"  {sd['name']}/ — {sd.get('file_count', 0)} files, {sd.get('dir_count', 0)} subdirs"
            if sd.get("sample_extensions"):
                detail += f", extensions: {sd['sample_extensions']}"
            lines.append(detail)
            if sd.get("subdirs"):
                for nested in sd["subdirs"]:
                    lines.append(f"    {nested['name']}/ — {nested.get('file_count', 0)} files")
            elif sd.get("subdir_names"):
                lines.append(f"    subdirs: {sd['subdir_names']}")

    # --- Touchstone: log.txt ---
    if "log_txt" in facts:
        log = facts["log_txt"]
        lines.append(f"\nlog.txt: {log.get('total_lines', '?')} data lines, {log.get('columns_per_row', '?')} columns per row")
        if "header_comment" in log:
            lines.append(f"  COLUMN NAMES (from header): {log['header_comment']}")
            lines.append(f"  USE THESE EXACT NAMES for 'columns' in the schema.")
        if "first_row_values" in log:
            lines.append(f"  First row values: {log['first_row_values']}")
        if "constant_columns" in log and log["constant_columns"]:
            lines.append(f"  Constant columns (indices): {log['constant_columns']}")

    # --- Touchstone: SPData ---
    if "spdata" in facts:
        sp = facts["spdata"]
        lines.append(f"\nSPData/: {sp['file_count']} files, extension={sp['extension']}, ports={sp.get('port_count', '?')}")
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

    # --- Tabular CSV files (e.g. in tabular/ subdir) ---
    if "tabular_csvs" in facts:
        lines.append("\nTabular CSV files (geometric/input parameters):")
        for tc in facts["tabular_csvs"]:
            lines.append(f"  {tc.get('path', tc.get('filename', '?'))}: {tc.get('row_count', '?')} rows")
            if tc.get("columns"):
                lines.append(f"    COLUMNS: {tc['columns']}")
                lines.append(f"    USE THESE EXACT column names for input features.")
            if tc.get("numeric_columns"):
                lines.append(f"    Numeric columns: {tc['numeric_columns']}")
            if tc.get("constant_columns"):
                lines.append(f"    Constant columns: {tc['constant_columns']}")
            if tc.get("value_ranges"):
                for col, vr in tc["value_ranges"].items():
                    lines.append(f"    {col}: min={vr['min']}, max={vr['max']}")
            if tc.get("id_column"):
                lines.append(f"    ID/name column: {tc['id_column']}")

    # --- Top-level CSV files ---
    if "csv_files" in facts:
        lines.append("\nTop-level CSV files:")
        for csv_info in facts["csv_files"]:
            lines.append(f"  {csv_info.get('filename', '?')}")
            if csv_info.get("type") == "tabular":
                lines.append(f"    Type: tabular, columns: {csv_info.get('columns', [])}, rows: {csv_info.get('row_count', '?')}")
            else:
                if csv_info.get("header"):
                    lines.append(f"    Header: {csv_info['header'][:120]}")
                if csv_info.get("parameter_keys"):
                    keys_str = ", ".join(f"{pk['key']}={pk['example_value']}" for pk in csv_info["parameter_keys"])
                    lines.append(f"    Parameters: {keys_str}")
                lines.append(f"    Data points per sample: {csv_info.get('data_rows_in_first_block', '?')}")
                if "value_min" in csv_info:
                    lines.append(f"    Value range: {csv_info['value_min']:.4e} to {csv_info['value_max']:.4e}")
                if csv_info.get("dynamic_range"):
                    lines.append(f"    Dynamic range: {csv_info['dynamic_range']:.0f}x {'(RECOMMEND log10 transform)' if csv_info.get('recommend_log10') else ''}")

    # --- Pickle files ---
    if "pkl_files" in facts:
        lines.append("\nPickle files (pre-packed arrays):")
        for pkl in facts["pkl_files"]:
            detail = f"  {pkl.get('path', pkl.get('filename', '?'))}: type={pkl.get('type', '?')}"
            if "shape" in pkl:
                detail += f", shape={pkl['shape']}"
            if "dtype" in pkl:
                detail += f", dtype={pkl['dtype']}"
            if "length" in pkl:
                detail += f", length={pkl['length']}"
            lines.append(detail)

    # --- Per-sample data directories ---
    if "per_sample_dirs" in facts:
        lines.append("\nPer-sample data directories:")
        for psd in facts["per_sample_dirs"]:
            lines.append(f"  {psd['path']}/: {psd['file_count']} files ({psd.get('dominant_extension', '?')})")
            if psd.get("csv_columns"):
                lines.append(f"    CSV columns: {psd['csv_columns']}")
                lines.append(f"    Rows per file: {psd.get('rows_per_file', '?')}")
            if psd.get("touchstone_header"):
                th = psd["touchstone_header"]
                lines.append(f"    Touchstone: freq_unit={th.get('frequency_unit', '?')}, format={th.get('data_format', '?')}")

    # --- Image directories ---
    if "image_dirs" in facts:
        lines.append("\nImage/layout directories:")
        for imd in facts["image_dirs"]:
            lines.append(f"  {imd['path']}/: {imd['file_count']} files ({imd.get('extension', '?')})")

    # --- Port info ---
    if "port_info" in facts:
        lines.append("\nPort info:")
        for pi in facts["port_info"]:
            lines.append(f"  {pi['path']}/: {pi['file_count']} files, columns: {pi['columns']}, example: {pi['example']}")

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
You are an AI assistant embedded in a surrogate-model training GUI.

The user has selected a dataset folder.  You have been given a detailed scan
of every file and directory inside it — column names, array shapes, row counts,
value ranges, and file previews.

Your job:
1. **Understand** what the user's dataset contains by reading the scan facts.
2. **Answer questions** about the dataset concisely.
3. **Generate a README.md** with a valid JSON schema block when asked, so the
   training pipeline can load the data automatically.

## How to identify file roles

Look at the scan facts to determine:
- **Input parameters**: A tabular file where each row is one sample and
  columns are design/geometric parameters.  Typically few columns, many rows,
  and a sample-ID column that links to the ground-truth files.
- **Ground truth**: Simulation output data the model will predict.  Could be
  per-sample CSV/touchstone files (many files, each with frequency-swept data),
  Cadence-style block CSVs, or pre-packed numpy/pickle arrays.
- **Layout**: Pixel/image representations of designs (PNG groups, GDS files,
  numpy arrays with spatial shapes).
- **Metadata**: Supporting data (port info, config files).

Use ONLY the column names and structure from the scan facts.  NEVER copy
column names or parameters from other datasets or examples.

## Supported ground-truth formats for the schema

The schema uses ``"source"`` (not ``"format"``) in the ground_truth block.
Three loading strategies are supported:

- ``"inline"``: single CSV with ALL samples.  Each sample spans multiple
  rows (one per frequency point), grouped by identical feature-column values.
  MUST set ``frequency_column``.  GT column names are auto-detected from
  common conventions.  For Cadence block-structured
  CSVs, use ``channel_files`` and ``input_feature.source: "embedded"``.
- ``"per_sample"``: one GT file per sample in ``data_dir/``.  File type is
  auto-detected from ``file_extension`` (``.csv``, ``.s2p``, ``.s4p``, etc.).
  Input features in a separate tabular CSV with ``sample_id_column``.
- ``"array"``: pre-packed numpy arrays in a pickle file.  Shape must be
  (num_samples, num_frequencies, num_channels).  Set ``data_dir`` to the
  pickle path.  MUST include ``frequency_start_hz`` / ``frequency_stop_hz``.

## S-parameter rules

For ground_truth_parameters, list the LOWER-TRIANGULAR unique S-parameters:
- 2-port: S11, S12, S22
- 4-port: S11, S12, S13, S14, S22, S23, S24, S33, S34, S44

ground_truth_parts MUST use: "re", "im", "mag", "db", or "angle_deg".

IMPORTANT: ground_truth_parameters + ground_truth_parts define the expected CSV
column names.  For example, parameters=["S11"] + parts=["re","im"] means the
loader will look for columns named "s11_real" and "s11_imag" (case-insensitive).
Do NOT include scalar columns (like R, L, Q, inductance, resistance) in
ground_truth_parameters — they don't have real/imaginary parts.  Only include
parameters whose CSV columns follow the {param}_{part} naming pattern.

## Generating loader.py (PREFERRED approach)

Instead of a README with a JSON schema, you should generate a **loader.py**
file that directly loads the data using Python.  This is more flexible and
handles any file format.

When the user asks to set up data loading, configure, or generate a loader,
output a COMPLETE loader.py inside a code block fenced with ```python ... ```.

The loader.py MUST define this function:

```
def load_dataset(dataset_root: str, max_samples: int | None = None) -> dict:
    """Load the dataset and return standardized arrays.

    Returns a dict with:
        features:      np.ndarray (num_samples, num_features)
        targets:        np.ndarray (num_samples, num_channels, num_sweep_points)
        sweep_axis:     np.ndarray (num_sweep_points,)
        feature_names:  list[str]
        channel_names:  list[str]
        channel_units:  list[str]   e.g. ["dB", "deg"]
        sweep_label:    str         e.g. "Frequency (GHz)" or "VDIFF (mV)"
        dataset_name:   str
    """
```

Rules for loader.py:
- Use ONLY: numpy, csv, re, json, pathlib, math, os.path, struct, io.
- Do NOT import pandas, scipy, torch, or any other libraries.
- Parse files using csv.DictReader, csv.reader, or manual line splitting.
- Handle SI suffixes in Cadence files (e.g. '80.98f' = 80.98e-15).
- targets must be shape (samples, channels, sweep_points).
- If max_samples is not None, only load that many samples.
- Include the dataset_name in the returned dict.

Example for a Cadence-style dataset:

```python
import re
import csv
import numpy as np
from pathlib import Path

_SI = {'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'K':1e3,'M':1e6,'G':1e9,'T':1e12}
def _parse_si(s):
    s = s.strip()
    if s and s[-1] in _SI: return float(s[:-1]) * _SI[s[-1]]
    return float(s)

def load_dataset(dataset_root, max_samples=None):
    root = Path(dataset_root)
    # ... parse files, extract features and targets ...
    return {
        "features": features,    # np.ndarray (N, num_features)
        "targets": targets,       # np.ndarray (N, num_channels, num_sweep_points)
        "sweep_axis": sweep,      # np.ndarray (num_sweep_points,)
        "feature_names": [...],
        "channel_names": [...],
        "channel_units": [...],
        "sweep_label": "Frequency (GHz)",
        "dataset_name": "MyDataset",
    }
```

## README generation (FALLBACK)

If the user specifically asks for a README with a JSON schema instead of
loader.py, output the COMPLETE README.md inside ```readme ... ```.

The README must contain:
1. A title and brief description
2. A fenced ```json``` block with a valid schema containing:
   - dataset_name
   - input_feature: columns, feature_columns, sample_id_column, file_path
   - ground_truth: source, ground_truth_parameters, ground_truth_parts,
     file_extension, drop_first_frequency, data_dir
3. Notes about the dataset

## Rules

- CRITICAL: Use ONLY column names from the scan facts.
- CRITICAL: input_feature MUST include BOTH `columns` (ALL columns in the file,
  in order) AND `feature_columns` (the subset used as model inputs).
  `columns` is REQUIRED — the parser will reject the schema without it.
- CRITICAL: Column names must be clean strings with NO brackets, quotes, or
  punctuation from the raw file format.  If the scan shows values like
  `[rax, ray, ...]` from a Python list, the column name is `rax` NOT `[rax`.
  Strip any leading `[` or trailing `]` from column names.
- When generating a README, output the COMPLETE content inside ```readme ... ```.
- If the user asks you to fix or change fields, output the full updated README.
- Keep answers concise and focused.
- If you need information to decide (e.g. which file is input vs output),
  ASK the user rather than guessing.
"""


class ChatSession:
    """Multi-turn conversation session with Gemini for dataset assistance.

    Uses ``dataset_scanner`` to analyze the dataset directory and provides
    the scan results as context for every conversation.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.history: list[dict] = []  # [{"role": "user"/"model", "text": str}]
        self._scan_context: str | None = None
        self._file_preview: str | None = None

    def set_dataset(self, dataset_dir: str | Path) -> str:
        """Scan the dataset directory. Returns a summary of what was found."""
        from .dataset_scanner import scan_directory, format_scan_for_llm

        self.dataset_dir = Path(dataset_dir)
        scan = scan_directory(str(self.dataset_dir), max_depth=2)
        self._scan_context = format_scan_for_llm(scan)
        self._file_preview = _sample_files(self.dataset_dir)
        return f"Dataset loaded: {self.dataset_dir.name}\n{len(scan.files)} files/groups found."

    def send(self, user_message: str) -> str:
        """Send a message and get a response. Blocking call."""
        try:
            from google import genai
        except ImportError:
            return "Error: google-genai package not installed. Run: pip install google-genai"

        if not self.api_key:
            return "Error: No Gemini API key configured. Set GOOGLE_API_KEY environment variable or enter it in Settings."

        # Auto-scan on first message if not done yet.
        if self._scan_context is None and self.dataset_dir and self.dataset_dir.is_dir():
            self.set_dataset(self.dataset_dir)

        # Build context block from scan results + file previews.
        context_block = ""
        if self._scan_context:
            context_block = (
                f"\n\n[Dataset scan results]\n"
                f"```\n{self._scan_context}\n```\n\n"
                f"[Sample file content]\n{self._file_preview}\n\n"
            )

        # Build contents for the API call (full history).
        contents = []
        for msg in self.history:
            contents.append(genai.types.Content(
                role=msg["role"],
                parts=[genai.types.Part(text=msg["text"])],
            ))

        # Attach context on first message, or when context hasn't been sent yet.
        enriched_message = user_message
        if not self.history and context_block:
            enriched_message = user_message + context_block
        elif context_block and not any("Dataset scan results" in m["text"] for m in self.history):
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

        self.history.append({"role": "user", "text": enriched_message})
        self.history.append({"role": "model", "text": reply})

        return reply

    def extract_readme(self, text: str) -> str | None:
        """Extract README content from a ```readme ... ``` fenced block.

        The README itself contains nested fenced blocks (```json ... ```), so
        we can't use a simple non-greedy match.  Instead, match the closing
        ``` that appears on its own line (not followed by a language tag).
        """
        # Strategy: find the opening fence, then scan for the matching close.
        for opener in ("```readme", "```markdown"):
            start = text.find(opener)
            if start == -1:
                continue
            # Skip past the opener line.
            content_start = text.find("\n", start)
            if content_start == -1:
                continue
            content_start += 1

            # Walk through lines to find the matching closing ```.
            # A closing ``` is a line that is exactly ``` (possibly with trailing whitespace),
            # NOT followed by a language tag (like ```json).
            depth = 0
            pos = content_start
            close_pos = None
            while pos < len(text):
                line_end = text.find("\n", pos)
                if line_end == -1:
                    line_end = len(text)
                line = text[pos:line_end].strip()
                if line.startswith("```") and len(line) > 3 and line[3:].strip().isalpha():
                    # Opening a nested fence (e.g. ```json, ```python).
                    depth += 1
                elif line == "```":
                    if depth > 0:
                        depth -= 1
                    else:
                        # This is our closing fence.
                        close_pos = pos
                        break
                pos = line_end + 1

            if close_pos is not None:
                return text[content_start:close_pos].strip()

        # Fallback: if the response contains a ```json block with "dataset_name",
        # treat the entire response as the README.
        if '"dataset_name"' in text and "```json" in text:
            return text.strip()
        return None

    def extract_loader(self, text: str) -> str | None:
        """Extract loader.py content from a ```python ... ``` fenced block.

        Only returns content if it contains 'def load_dataset'.
        """
        pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
        for match in pattern.finditer(text):
            code = match.group(1).strip()
            if "def load_dataset" in code:
                return code
        return None

    def clear(self) -> None:
        """Reset conversation history."""
        self.history.clear()
