"""Convert an Axion .platemap file into a simple well-to-group CSV.

The output CSV is used by the MEA first-pass pipeline. It contains:

well,group,include,note,raw_label

The parser extracts 48 row-major labels from the binary .platemap file. The
result must be reviewed before running group analysis, because group names come
from the Axion plate map labels.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path


DEFAULT_WELL_ROWS = 6
DEFAULT_WELL_COLS = 8


def extract_length_prefixed_ascii(path: Path) -> list[str]:
    data = path.read_bytes()
    candidates: list[tuple[int, int, str]] = []

    for offset in range(0, max(0, len(data) - 4)):
        length = struct.unpack_from("<I", data, offset)[0]
        if not 1 <= length <= 120:
            continue

        start = offset + 4
        end = start + length
        if end > len(data):
            continue

        raw = data[start:end]
        if not all(32 <= value <= 126 for value in raw):
            continue

        text = raw.decode("ascii", errors="ignore").strip()
        if not text:
            continue
        if text == "AxionBio":
            continue
        if text.lower().startswith("started on"):
            continue

        candidates.append((offset, length, text))

    labels: list[str] = []
    previous_offset = -1
    previous_length = -1
    previous_text = ""

    for offset, length, text in candidates:
        # In Axion .platemap files, Empty wells can appear twice in a row:
        # one entry for the label and one for the note. Skip the immediate
        # duplicate note entry.
        immediate_duplicate = (
            text == previous_text
            and offset == previous_offset + 4 + previous_length
        )
        if immediate_duplicate:
            previous_offset = offset
            previous_length = length
            previous_text = text
            continue

        labels.append(text)
        previous_offset = offset
        previous_length = length
        previous_text = text

    return labels


def clean_group(raw_label: str) -> str:
    label = raw_label.strip()
    if not label:
        return "Unknown"
    if label.lower() == "empty":
        return "Empty"

    # Remove common trailing free-text fragments if they exist.
    label = re.sub(r"\s*\(.*$", "", label).strip()
    label = re.sub(r"\s+", " ", label)
    return label or "Unknown"


def well_name(row_index: int, col_index: int) -> str:
    return f"{chr(ord('A') + row_index)}{col_index + 1}"


def convert_platemap(
    input_path: Path,
    output_path: Path,
    well_rows: int = DEFAULT_WELL_ROWS,
    well_cols: int = DEFAULT_WELL_COLS,
) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到 .platemap 文件: {input_path}")

    labels = extract_length_prefixed_ascii(input_path)
    expected = well_rows * well_cols
    if len(labels) != expected:
        raise ValueError(
            f"解析到 {len(labels)} 个 well labels，但预期为 {expected} 个。\n"
            "请检查 .platemap 文件是否对应 48-well Axion plate，"
            "或手动整理 CSV。\n"
            f"解析结果: {labels}"
        )

    rows: list[dict[str, str]] = []
    for row_index in range(well_rows):
        for col_index in range(well_cols):
            raw_label = labels[row_index * well_cols + col_index]
            group = clean_group(raw_label)
            include = "0" if group.lower() == "empty" else "1"
            note = "empty well" if include == "0" else ""
            rows.append(
                {
                    "well": well_name(row_index, col_index),
                    "group": group,
                    "include": include,
                    "note": note,
                    "raw_label": raw_label,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["well", "group", "include", "note", "raw_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1

    print("\n=== Converted plate map summary ===")
    print(f"Output CSV: {output_path.resolve()}")
    print("Group counts:")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")
    print("\nPlate layout:")
    for row_index in range(DEFAULT_WELL_ROWS):
        row = rows[row_index * DEFAULT_WELL_COLS : (row_index + 1) * DEFAULT_WELL_COLS]
        print("  " + " | ".join(f"{item['well']}={item['group']}" for item in row))

    print(
        "\n请打开输出 CSV，确认 group 名称、Empty wells 和 control group 是否正确。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Axion .platemap to a well/group CSV file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Axion .platemap file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = convert_platemap(args.input, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
