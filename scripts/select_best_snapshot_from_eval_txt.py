"""
Select the best checkpoint from a txt file of snapshot evaluation results.

The script parses snapshot result blocks before the optional "Results Summary"
section, extracts BLEU/METEOR/ROUGE-L/CIDEr/SPICE metrics, and selects one
checkpoint with the same rule for baseline and ours:

    Score = Rank(CIDEr) + Rank(SPICE) + 0.5 * Rank(METEOR)
            + 0.3 * Rank(BLEU4)

All metrics are treated as larger-is-better. Ranks are computed among the
parsed snapshots in the same txt file after optional checkpoint filtering.
Lower score is better. If stable_window > 1, the final selection uses a
rolling mean of score over checkpoints sorted in ascending order.
"""

import argparse
import os
import re
import sys

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Error: pandas is required. Install project dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


HEADER_RE = re.compile(r"^\s*={5,}\s*(.*?)\s+results\s*={5,}\s*$", re.IGNORECASE)
SUMMARY_RE = re.compile(r"Results\s+Summary", re.IGNORECASE)
NUMBER_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

METRIC_COLUMNS = [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
    "SPICE",
]

RANK_COLUMNS = [
    "rank_CIDEr",
    "rank_SPICE",
    "rank_METEOR",
    "rank_Bleu_4",
]

METRIC_PATTERNS = {
    "Bleu_1": re.compile(r"^\s*(?:BLEU[\s_-]?1|B1)\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "Bleu_2": re.compile(r"^\s*(?:BLEU[\s_-]?2|B2)\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "Bleu_3": re.compile(r"^\s*(?:BLEU[\s_-]?3|B3)\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "Bleu_4": re.compile(r"^\s*(?:BLEU[\s_-]?4|B4)\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "METEOR": re.compile(r"^\s*METEOR\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "ROUGE_L": re.compile(r"^\s*ROUGE[\s_-]?L\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "CIDEr": re.compile(r"^\s*CIDEr(?:-D)?\s*:\s*" + NUMBER_RE, re.IGNORECASE),
    "SPICE": re.compile(r"^\s*SPICE\s*:\s*" + NUMBER_RE, re.IGNORECASE),
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value for --save_all.")


def warn(message):
    print("Warning: %s" % message, file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best checkpoint from txt snapshot evaluation results."
    )
    parser.add_argument("--input", required=True, help="Input eval-result txt file path.")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("results", "snapshot_selection_txt"),
        help="Directory for best_checkpoint.csv, best_checkpoint.md, and optional all scores.",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        help="Model config name. If omitted, infer it from each snapshot name.",
    )
    parser.add_argument("--min_ckpt", type=int, default=None, help="Minimum checkpoint to keep.")
    parser.add_argument("--max_ckpt", type=int, default=None, help="Maximum checkpoint to keep.")
    parser.add_argument(
        "--stable_window",
        type=int,
        default=1,
        help="Rolling score window over checkpoints. Use 1 to disable.",
    )
    parser.add_argument(
        "--save_all",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Save all_checkpoint_scores.csv. Supports --save_all or --save_all=True.",
    )
    return parser.parse_args()


def read_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Input file does not exist: %s" % path)

    encodings = ["utf-8-sig", "utf-8", "gbk", "cp936"]
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        "Could not decode input file with utf-8-sig, utf-8, gbk, or cp936.",
    )


def split_snapshot_blocks(text):
    blocks = []
    current_header = None
    current_lines = []

    for line in text.splitlines():
        if SUMMARY_RE.search(line):
            break

        if HEADER_RE.match(line):
            if current_header is not None:
                blocks.append((current_header, "\n".join(current_lines)))
            current_header = line.strip()
            current_lines = []
            continue

        if current_header is not None:
            current_lines.append(line)

    if current_header is not None:
        blocks.append((current_header, "\n".join(current_lines)))

    return blocks


def parse_snapshot_header(header):
    match = HEADER_RE.match(header)
    if not match:
        return None

    snapshot_name = match.group(1).strip()
    checkpoint_match = re.search(r"(\d+)\s*$", snapshot_name)
    if not checkpoint_match:
        return None

    checkpoint = int(checkpoint_match.group(1))
    config_end = checkpoint_match.start()
    if config_end > 0 and snapshot_name[config_end - 1] == "_":
        config_end -= 1

    return {
        "snapshot_name": snapshot_name,
        "checkpoint": checkpoint,
        "config_name": snapshot_name[:config_end],
    }


def parse_metrics_from_block(block_text):
    metrics = {}

    for line in block_text.splitlines():
        for metric_name, pattern in METRIC_PATTERNS.items():
            if metric_name in metrics:
                continue
            match = pattern.match(line)
            if match:
                metrics[metric_name] = float(match.group(1))

    return metrics


def build_result_dataframe(blocks, source_file, config_name=None):
    rows = []

    for header, block_text in blocks:
        header_info = parse_snapshot_header(header)
        if header_info is None:
            warn("Skipping block with unrecognized snapshot header: %s" % header)
            continue

        snapshot_name = header_info["snapshot_name"]
        metrics = parse_metrics_from_block(block_text)

        if "CIDEr" not in metrics or "SPICE" not in metrics:
            warn(
                "Skipping %s because CIDEr or SPICE is missing. These are primary metrics."
                % snapshot_name
            )
            continue

        for optional_metric in ["METEOR", "Bleu_4"]:
            if optional_metric not in metrics:
                warn(
                    "%s is missing %s; its score contribution will be skipped for this row."
                    % (snapshot_name, optional_metric)
                )

        row = {
            "config_name": config_name or header_info["config_name"],
            "checkpoint": header_info["checkpoint"],
            "snapshot_name": snapshot_name,
            "source_file": source_file,
        }
        for metric_name in METRIC_COLUMNS:
            row[metric_name] = metrics.get(metric_name, pd.NA)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No valid snapshot blocks were parsed. Check header format, for example: "
            "====================xxx_8000 results===================="
        )

    df = pd.DataFrame(rows)
    for metric_name in METRIC_COLUMNS:
        df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    df["checkpoint"] = pd.to_numeric(df["checkpoint"], errors="raise").astype(int)
    return df


def compute_metric_ranks(df):
    df = df.copy()
    rank_map = {
        "CIDEr": "rank_CIDEr",
        "SPICE": "rank_SPICE",
        "METEOR": "rank_METEOR",
        "Bleu_4": "rank_Bleu_4",
    }

    for metric_name, rank_name in rank_map.items():
        df[rank_name] = df[metric_name].rank(method="min", ascending=False)

    return df


def compute_selection_score(df):
    df = df.copy()
    df["score"] = (
        df["rank_CIDEr"]
        + df["rank_SPICE"]
        + 0.5 * df["rank_METEOR"].fillna(0.0)
        + 0.3 * df["rank_Bleu_4"].fillna(0.0)
    )
    return df


def apply_stable_score(df, stable_window):
    if stable_window < 1:
        raise ValueError("--stable_window must be >= 1.")
    if stable_window == 1:
        return df

    df = df.copy()
    sorted_index = df.sort_values(["checkpoint", "snapshot_name"]).index
    stable_scores = (
        df.loc[sorted_index, "score"]
        .rolling(window=stable_window, min_periods=1, center=True)
        .mean()
    )
    df.loc[sorted_index, "stable_score"] = stable_scores.to_numpy()
    return df


def select_best_checkpoint(df):
    selection_column = "stable_score" if "stable_score" in df.columns else "score"
    sorted_df = df.sort_values(
        [selection_column, "SPICE", "CIDEr", "METEOR", "Bleu_4", "checkpoint"],
        ascending=[True, False, False, False, False, False],
        na_position="last",
    )
    return sorted_df.iloc[0].copy()


def dataframe_to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = list(df.columns)
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            values = ["" if pd.isna(row[col]) else str(row[col]) for col in headers]
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)


def save_outputs(best_row, all_df, output_dir, save_all):
    os.makedirs(output_dir, exist_ok=True)

    has_stable_score = "stable_score" in all_df.columns
    best_columns = ["config_name", "selected_checkpoint", "snapshot_name"]
    best_columns += METRIC_COLUMNS + RANK_COLUMNS + ["score"]
    if has_stable_score:
        best_columns.append("stable_score")
    best_columns.append("source_file")

    all_columns = ["config_name", "checkpoint", "snapshot_name"]
    all_columns += METRIC_COLUMNS + RANK_COLUMNS + ["score"]
    if has_stable_score:
        all_columns.append("stable_score")
    all_columns.append("source_file")

    best_df = pd.DataFrame([best_row])
    best_df["selected_checkpoint"] = best_df["checkpoint"].astype(int)
    best_df = best_df[best_columns]

    best_csv_path = os.path.join(output_dir, "best_checkpoint.csv")
    best_md_path = os.path.join(output_dir, "best_checkpoint.md")
    all_csv_path = os.path.join(output_dir, "all_checkpoint_scores.csv")

    best_df.to_csv(best_csv_path, index=False)
    with open(best_md_path, "w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(best_df))
        f.write("\n")

    if save_all:
        all_df = all_df.sort_values(["score", "checkpoint"], ascending=[True, True])
        all_df[all_columns].to_csv(all_csv_path, index=False)

    return {
        "best_csv": best_csv_path,
        "best_md": best_md_path,
        "all_csv": all_csv_path if save_all else None,
    }


def main():
    args = parse_args()
    source_file = os.path.abspath(args.input)
    text = read_txt_file(source_file)
    blocks = split_snapshot_blocks(text)
    if not blocks:
        raise RuntimeError(
            "No snapshot block was found. Check header format, for example: "
            "====================xxx_8000 results===================="
        )

    df = build_result_dataframe(blocks, source_file, config_name=args.config_name)
    if args.min_ckpt is not None:
        df = df[df["checkpoint"] >= args.min_ckpt]
    if args.max_ckpt is not None:
        df = df[df["checkpoint"] <= args.max_ckpt]
    if df.empty:
        raise RuntimeError("No checkpoint remains after min_ckpt/max_ckpt filtering.")

    df = compute_metric_ranks(df)
    df = compute_selection_score(df)
    df = apply_stable_score(df, args.stable_window)

    best_row = select_best_checkpoint(df)
    output_paths = save_outputs(best_row, df, args.output_dir, args.save_all)

    print("[%s]" % best_row["config_name"])
    print("Selected checkpoint: %d" % int(best_row["checkpoint"]))
    print("Snapshot name: %s" % best_row["snapshot_name"])
    print("Score: %s" % best_row["score"])
    if "stable_score" in df.columns:
        print("Stable score: %s" % best_row["stable_score"])
    print("CIDEr: %s" % best_row["CIDEr"])
    print("SPICE: %s" % best_row["SPICE"])
    print("Bleu_4: %s" % best_row["Bleu_4"])
    print("METEOR: %s" % best_row["METEOR"])
    print("Source file: %s" % best_row["source_file"])
    print("Saved best CSV: %s" % output_paths["best_csv"])
    print("Saved best Markdown: %s" % output_paths["best_md"])
    if output_paths["all_csv"] is not None:
        print("Saved all scores CSV: %s" % output_paths["all_csv"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("Error: %s" % exc)
