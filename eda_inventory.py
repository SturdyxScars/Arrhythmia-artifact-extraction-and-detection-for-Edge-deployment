"""Create a record-level and lead-level EDA inventory for MIT-BIH.

Run from the project root:
    python .\eda_inventory.py

The script reads the canonical WFDB records named by their .hea files. It
prints progress and writes three CSV files next to this script:
    record_inventory.csv  one row per ECG record
    lead_inventory.csv    one row per signal/channel
    annotation_counts.csv one row per record and annotation category

The data-quality flags are screening summaries, not automatic exclusions.
Review the ECG plots before deciding that unusual values are artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import wfdb


# Paths are relative to this script, so it can be run from another working
# directory as well as from the project root.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "mit-bih-arrhythmia-database-1.0.0"

# WFDB's MIT-BIH beat symbols, grouped using a commonly used AAMI mapping.
# This mapping is a declared starting convention; review it for your project
# and keep the exact mapping with any training/evaluation results.
AAMI_GROUPS: dict[str, str] = {
    **{symbol: "N" for symbol in ("N", "L", "R", "e", "j")},
    **{symbol: "S" for symbol in ("A", "a", "J", "S")},
    **{symbol: "V" for symbol in ("V", "E")},
    "F": "F",
    **{symbol: "Q" for symbol in ("/", "f", "Q", "?")},
}

# These are annotation codes, not beat classes. WFDB uses '+' for rhythm
# changes and '~' for signal-quality changes in MIT-BIH annotations.
RHYTHM_SYMBOL = "+"
QUALITY_SYMBOL = "~"

# A run this long with exactly the same calibrated sample is worth flagging.
# It is only a heuristic: quantization can produce short repeated runs in
# healthy data, and a flatline can also be an acquisition fault.
FLATLINE_RUN_SECONDS = 1.0


def json_text(value: Any) -> str:
    """Serialize lists/dictionaries consistently for storage in a CSV cell."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def longest_equal_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of identical samples."""
    if values.size == 0:
        return 0
    # A change marks the start of a new run. The sentinel at the end closes
    # the final run, including when the complete signal is constant.
    change_points = np.flatnonzero(values[1:] != values[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [values.size]))
    return int(np.diff(boundaries).max())


def annotation_category(symbol: str) -> str:
    """Classify annotation symbols without treating every symbol as a beat."""
    if symbol in AAMI_GROUPS:
        return "beat"
    if symbol == RHYTHM_SYMBOL:
        return "rhythm"
    if symbol == QUALITY_SYMBOL:
        return "quality"
    # Remaining entries include comments, waveform markers, and other
    # database-specific annotations. Keep their symbol counts for inspection.
    return "other_comment_or_marker"


def inspect_record(record_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read one record and return its record, channel, and annotation rows."""
    base = DATA_DIR / record_id
    expected_files = {
        "header": base.with_suffix(".hea"),
        "signal": base.with_suffix(".dat"),
        # .atr is the canonical reference annotation named in ANNOTATORS.
        "annotation": base.with_suffix(".atr"),
    }
    missing_files = [kind for kind, path in expected_files.items() if not path.is_file()]

    record_row: dict[str, Any] = {
        "record_id": record_id,
        "header_path": str(expected_files["header"].relative_to(HERE)),
        "missing_files": json_text(missing_files),
        "sampling_hz": "",
        "sample_count": "",
        "duration_seconds": "",
        "channel_count": "",
        "channel_names": "",
        "channel_units": "",
        "annotation_file": str(expected_files["annotation"].relative_to(HERE)),
        "annotation_loaded": False,
        "annotation_count": 0,
        "annotation_sample_min": "",
        "annotation_sample_max": "",
        "annotation_time_min_seconds": "",
        "annotation_time_max_seconds": "",
        "annotation_out_of_range_count": "",
        "beat_symbol_counts": "{}",
        "aami_beat_counts": "{}",
        "rhythm_label_counts": "{}",
        "quality_label_counts": "{}",
        "other_symbol_counts": "{}",
        "read_error": "",
    }
    lead_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []

    # Report missing components while still attempting to read whatever is
    # available. WFDB will raise a useful exception if required files are absent.
    try:
        rec = wfdb.rdrecord(str(base), physical=True)
        signal = np.asarray(rec.p_signal)
        fs = float(rec.fs)
        n_samples, n_channels = signal.shape

        record_row.update(
            sampling_hz=fs,
            sample_count=n_samples,
            duration_seconds=n_samples / fs if fs else "",
            channel_count=n_channels,
            channel_names=json_text(list(rec.sig_name)),
            channel_units=json_text(list(rec.units)),
        )

        # Use digital values and the ADC metadata to flag samples at the
        # representable range boundaries. This is a screening heuristic, not
        # proof of physiological clipping.
        digital = np.asarray(rec.d_signal) if rec.d_signal is not None else None

        for channel_index, channel_name in enumerate(rec.sig_name):
            x = signal[:, channel_index].astype(np.float64, copy=False)
            finite_mask = np.isfinite(x)
            finite = x[finite_mask]

            if finite.size:
                median = float(np.median(finite))
                mad = float(np.median(np.abs(finite - median)))
                q25, q75 = np.percentile(finite, [25, 75])
                longest_run = longest_equal_run(finite)
                flatline_samples = int(round(FLATLINE_RUN_SECONDS * fs))

                rail_fraction: float | str = ""
                if digital is not None:
                    d = digital[:, channel_index]
                    adc_res = int(rec.adc_res[channel_index])
                    adc_zero = int(rec.adc_zero[channel_index])
                    low_rail = adc_zero - (2 ** (adc_res - 1))
                    high_rail = adc_zero + (2 ** (adc_res - 1)) - 1
                    at_rail = (d <= low_rail) | (d >= high_rail)
                    rail_fraction = float(np.mean(at_rail))

                stats = {
                    "minimum": float(np.min(finite)),
                    "maximum": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "median": median,
                    "mad": mad,
                    "iqr": float(q75 - q25),
                    "non_finite_count": int(x.size - finite.size),
                    "longest_equal_run_samples": longest_run,
                    "longest_equal_run_seconds": longest_run / fs if fs else "",
                    "flatline_over_1s_flag": bool(longest_run >= flatline_samples),
                    "adc_rail_fraction_heuristic": rail_fraction,
                }
            else:
                stats = {
                    "minimum": "", "maximum": "", "mean": "", "std": "",
                    "median": "", "mad": "", "iqr": "",
                    "non_finite_count": int(x.size),
                    "longest_equal_run_samples": "",
                    "longest_equal_run_seconds": "",
                    "flatline_over_1s_flag": False,
                    "adc_rail_fraction_heuristic": "",
                }

            lead_rows.append({
                "record_id": record_id,
                "channel_index": channel_index,
                "channel_name": channel_name,
                "unit": rec.units[channel_index],
                "sampling_hz": fs,
                "sample_count": n_samples,
                **stats,
            })

        # Read only the canonical .atr file. Alternate files such as 102-0.atr
        # remain visible in the directory but are not silently substituted.
        if expected_files["annotation"].is_file():
            ann = wfdb.rdann(str(base), "atr")
            samples = np.asarray(ann.sample, dtype=np.int64)
            symbols = list(ann.symbol)
            aux_notes = list(ann.aux_note)
            counts_by_category: dict[str, Counter[str]] = {
                "beat": Counter(), "rhythm": Counter(), "quality": Counter(),
                "other_comment_or_marker": Counter(),
            }
            aami_counts: Counter[str] = Counter()

            for i, symbol in enumerate(symbols):
                category = annotation_category(symbol)
                counts_by_category[category][symbol] += 1
                if category == "beat":
                    aami_counts[AAMI_GROUPS[symbol]] += 1
                aux = aux_notes[i] if i < len(aux_notes) else ""
                annotation_rows.append({
                    "record_id": record_id,
                    "category": category,
                    "symbol": symbol,
                    "aami_group": AAMI_GROUPS.get(symbol, ""),
                    "aux_note": aux,
                    "count": 1,
                })

            rhythm_labels = Counter(
                note for note in aux_notes
                if note and (note.startswith("(") or note.startswith("["))
            )
            comment_labels = Counter(
                aux_notes[i] for i, symbol in enumerate(symbols)
                if annotation_category(symbol) == "other_comment_or_marker"
                and i < len(aux_notes) and aux_notes[i]
            )

            # Out-of-range samples include negative indices and indices at or
            # beyond signal length. Annotation sample values are zero-based.
            out_of_range = (samples < 0) | (samples >= n_samples)
            valid_samples = samples[~out_of_range]
            record_row.update(
                annotation_loaded=True,
                annotation_count=len(samples),
                annotation_sample_min=int(samples.min()) if samples.size else "",
                annotation_sample_max=int(samples.max()) if samples.size else "",
                annotation_time_min_seconds=float(samples.min() / fs) if samples.size and fs else "",
                annotation_time_max_seconds=float(samples.max() / fs) if samples.size and fs else "",
                annotation_out_of_range_count=int(np.count_nonzero(out_of_range)),
                beat_symbol_counts=json_text(dict(counts_by_category["beat"])),
                aami_beat_counts=json_text(dict(aami_counts)),
                rhythm_label_counts=json_text(dict(rhythm_labels)),
                quality_label_counts=json_text(dict(counts_by_category["quality"])),
                other_symbol_counts=json_text(dict(counts_by_category["other_comment_or_marker"])),
            )

            # Add a record-by-symbol total for annotations carrying auxiliary
            # comments, while retaining the distinct marker counts above.
            for label, count in comment_labels.items():
                annotation_rows.append({
                    "record_id": record_id,
                    "category": "aux_comment_text",
                    "symbol": "",
                    "aami_group": "",
                    "aux_note": label,
                    "count": count,
                })

            # A timestamp range based on valid annotations is useful when an
            # annotation is invalid; keep the raw min/max fields as diagnostics.
            if samples.size and not valid_samples.size:
                record_row["annotation_time_min_seconds"] = ""
                record_row["annotation_time_max_seconds"] = ""

    except Exception as exc:  # Keep inventory generation going for other records.
        record_row["read_error"] = f"{type(exc).__name__}: {exc}"

    return record_row, lead_rows, annotation_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries, including a header for an empty table."""
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    else:
        fieldnames = []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Enumerate headers (records), inspect each, and save inventory tables."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    # Headers define records. This avoids mistaking alternate annotation files
    # or other support files for additional ECG recordings.
    record_ids = sorted(
        (path.stem for path in DATA_DIR.glob("*.hea") if path.stem.isdigit()),
        key=int,
    )
    if not record_ids:
        raise FileNotFoundError(f"No numeric .hea record headers found in {DATA_DIR}")

    record_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []

    for index, record_id in enumerate(record_ids, start=1):
        row, channels, annotations = inspect_record(record_id)
        record_rows.append(row)
        lead_rows.extend(channels)
        annotation_rows.extend(annotations)
        status = "OK" if not row["read_error"] else "READ ERROR"
        print(f"[{index:02d}/{len(record_ids)}] {record_id}: {status}")

    write_csv(HERE / "record_inventory.csv", record_rows)
    write_csv(HERE / "lead_inventory.csv", lead_rows)
    write_csv(HERE / "annotation_counts.csv", annotation_rows)

    print("\nInventory complete")
    print(f"  Records discovered from headers: {len(record_ids)}")
    print(f"  Record table:      {HERE / 'record_inventory.csv'}")
    print(f"  Lead table:        {HERE / 'lead_inventory.csv'}")
    print(f"  Annotation table:  {HERE / 'annotation_counts.csv'}")
    print("Review missing_files/read_error and all heuristic quality flags before analysis.")


if __name__ == "__main__":
    main()
