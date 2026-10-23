from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, List, Sequence


DEFAULT_INCSV = ""
DEFAULT_OUTCSV = ""


TEXT_COLS = ("video_id", "reference", "prediction")
ALIGNSCORE_COLUMN = "AlignScore"


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if str(value).lower() == "nan":
        return ""
    return str(value).strip()


def _as_float_list(values: Sequence, expected_len: int, metric_name: str) -> List[float]:
    vals = [float(v) if v is not None else math.nan for v in list(values)]
    if len(vals) != expected_len:
        raise RuntimeError(
            f"{metric_name} returned {len(vals)} scores for {expected_len} rows."
        )
    return vals


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _patch_transformers_adamw() -> bool:

    try:
        import transformers
        if hasattr(transformers, "AdamW"):
            return False
        import torch
        transformers.AdamW = torch.optim.AdamW
        return True
    except Exception:
        return False


def _patch_missing_pkg_resources() -> bool:

    if "pkg_resources" in sys.modules:
        return False
    if importlib.util.find_spec("pkg_resources") is not None:
        return False

    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.declare_namespace = lambda name: None
    sys.modules["pkg_resources"] = pkg_resources
    return True


def _alignscore_dependency_message(exc: Exception) -> str:
    torch_version = _package_version("torch")
    transformers_version = _package_version("transformers")
    setuptools_version = _package_version("setuptools")
    pytorch_lightning_version = _package_version("pytorch-lightning")
    exc_name = getattr(exc, "name", "") or ""
    exc_text = str(exc)

    hints = []
    if exc_name == "pkg_resources" or "pkg_resources" in exc_text:
        hints.append(
            "Missing `pkg_resources`: this comes from setuptools. "
            "Either keep the default eval_fac.py compatibility patch enabled, "
            "or install an older setuptools build, for example:\n\n"
            "  pip install 'setuptools<81'\n"
        )
    if "AdamW" in exc_text:
        hints.append(
            "Missing `transformers.AdamW`: alignscore 0.1.x expects older "
            "transformers. Use:\n\n"
            "  pip install 'transformers==4.30.2' 'tokenizers<0.14'\n"
        )
    if "PyTorch >= 2.1" in exc_text or "torch < 2.1" in exc_text:
        hints.append(
            "The installed transformers build requires torch >= 2.1. For this "
            "torch 1.13.x AlignScore environment, prefer transformers 4.30.2."
        )
    if not hints:
        hints.append(
            "For this AlignScore-only environment, a usually compatible stack is:\n\n"
            "  pip install 'transformers==4.30.2' 'tokenizers<0.14' 'setuptools<81'\n"
        )

    return (
        "AlignScore is installed, but its dependencies are incompatible with "
        "the current environment.\n\n"
        f"Original import error:\n  {type(exc).__name__}: {exc}\n\n"
        "Detected versions:\n"
        f"  torch             : {torch_version}\n"
        f"  transformers      : {transformers_version}\n"
        f"  setuptools        : {setuptools_version}\n"
        f"  pytorch-lightning : {pytorch_lightning_version}\n\n"
        "Suggested fix:\n"
        + "\n\n".join(hints)
        + "\n\n"
        "Then rerun eval_fac.py."
    )


def load_eval_csv(path: str, limit: int = 0) -> pd.DataFrame:
    import pandas as pd

    df = pd.read_csv(path)
    missing = [c for c in TEXT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df[df["video_id"].astype(str).str.upper() != "AVERAGE"].copy()
    if limit and limit > 0:
        df = df.iloc[:limit].copy()

    if len(df) == 0:
        raise ValueError("No evaluation rows found after removing AVERAGE/limit.")
    return df.reset_index(drop=True)


def save_with_average(df: pd.DataFrame, out_csv: str) -> None:
    import pandas as pd

    final_cols = list(df.columns)
    numeric_cols = []
    for col in final_cols:
        if col in TEXT_COLS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            df[col] = numeric
            numeric_cols.append(col)

    avg = {col: "" for col in final_cols}
    avg["video_id"] = "AVERAGE"
    avg["reference"] = f"N={len(df)}"
    avg["prediction"] = ""
    for col in numeric_cols:
        avg[col] = float(pd.to_numeric(df[col], errors="coerce").mean())

    df_out = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_alignscore_scorer(
    args: argparse.Namespace,
) -> Any:
    if importlib.util.find_spec("alignscore") is None:
        raise RuntimeError(
            "AlignScore is not installed. Install it from "
            "https://github.com/yuh-zha/AlignScore and run "
            "`python -m spacy download en_core_web_sm`."
        )

    if args.alignscore_patch_adamw:
        patched = _patch_transformers_adamw()
        if patched:
            print("[AlignScore] patched missing transformers.AdamW with torch.optim.AdamW")
    if args.alignscore_patch_pkg_resources:
        patched = _patch_missing_pkg_resources()
        if patched:
            print("[AlignScore] patched missing pkg_resources.declare_namespace")

    try:
        from alignscore import AlignScore
    except ImportError as exc:
        raise RuntimeError(_alignscore_dependency_message(exc)) from exc

    ckpt_path = args.alignscore_ckpt or os.environ.get("ALIGNSCORE_CKPT")
    if not ckpt_path:
        raise RuntimeError(
            "AlignScore checkpoint is required. Pass --alignscore_ckpt or set "
            "ALIGNSCORE_CKPT. For example: models/factuality/AlignScore-base.ckpt"
        )
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"AlignScore checkpoint not found: {ckpt_path}")

    print(
        f"[AlignScore] model={args.alignscore_model} "
        f"mode={args.alignscore_eval_mode} ckpt={ckpt_path}"
    )
    scorer = AlignScore(
        model=args.alignscore_model,
        batch_size=args.alignscore_batch_size,
        device=args.device,
        ckpt_path=ckpt_path,
        evaluation_mode=args.alignscore_eval_mode,
    )
    return scorer


def compute_alignscore(
    refs: Sequence[str],
    preds: Sequence[str],
    scorer: Any,
) -> List[float]:
    scores = scorer.score(contexts=list(refs), claims=list(preds))
    return _as_float_list(scores, len(refs), "AlignScore")


def metric_refs_and_preds(df: pd.DataFrame, indices: Sequence[int]) -> tuple[List[str], List[str]]:
    refs = [_safe_text(df.at[i, "reference"]) for i in indices]
    preds = [_safe_text(df.at[i, "prediction"]) for i in indices]
    return refs, preds


def append_alignscore(
    df: pd.DataFrame,
    args: argparse.Namespace,
    scorer: Any | None,
) -> Any | None:
    import pandas as pd

    col = ALIGNSCORE_COLUMN
    if col not in df.columns:
        df[col] = math.nan

    numeric_existing = pd.to_numeric(df[col], errors="coerce")
    todo_mask = pd.Series(args.overwrite, index=df.index)
    if not args.overwrite:
        todo_mask = numeric_existing.isna()
    todo_idx = df.index[todo_mask].tolist()

    if not todo_idx:
        print(f"[{col}] already present for all rows; skipping.")
        return scorer

    refs, preds = metric_refs_and_preds(df, todo_idx)
    try:
        if scorer is None:
            scorer = load_alignscore_scorer(args)
        scores = compute_alignscore(refs, preds, scorer)
    except Exception as exc:
        if not args.allow_missing:
            raise
        print(f"[WARN] {col} failed; filling NaN because --allow_missing was set.")
        print(f"       {exc}", file=sys.stderr)
        scores = [math.nan] * len(todo_idx)

    for idx, score in zip(todo_idx, scores):
        df.at[idx, col] = score
    print(f"[{col}] added for {len(todo_idx)} rows.")
    return scorer


def collect_csv_jobs(args: argparse.Namespace) -> List[tuple[Path, Path]]:
    if not args.in_dir:
        return [(Path(args.in_csv), Path(args.out_csv))]

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise ValueError(f"Input directory not found: {in_dir}")

    csv_paths = sorted(
        (
            path
            for path in in_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == ".csv"
                and not path.name.startswith("._")
            )
        ),
        key=lambda path: path.name.lower(),
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in input directory: {in_dir}")

    out_dir = Path(args.out_dir)
    return [
        (in_csv, out_dir / f"{in_csv.stem}{args.output_suffix}.csv")
        for in_csv in csv_paths
    ]


def evaluate_csv(
    in_csv: Path,
    out_csv: Path,
    args: argparse.Namespace,
    scorer: Any | None,
) -> Any | None:
    df = load_eval_csv(str(in_csv), limit=args.limit)
    print(f"Loaded {len(df)} rows from {in_csv}")

    scorer = append_alignscore(df, args, scorer)

    save_with_average(df, str(out_csv))
    print(f"[Done] wrote {out_csv}")

    import pandas as pd

    mean_val = pd.to_numeric(df[ALIGNSCORE_COLUMN], errors="coerce").mean()
    if pd.isna(mean_val):
        print(f"  {ALIGNSCORE_COLUMN:18s}: NaN")
    else:
        print(f"  {ALIGNSCORE_COLUMN:18s}: {mean_val:.4f}")
    return scorer


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--in_csv", type=str, default=DEFAULT_INCSV)
    input_group.add_argument(
        "--in_dir",
        type=str,
        default="",
        help="Process every CSV directly inside this directory.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--out_csv", type=str, default=DEFAULT_OUTCSV)
    output_group.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory used with --in_dir.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_fac",
        help="Suffix for directory-mode output filenames. Default: _fac",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Continue and fill NaN if AlignScore or its checkpoint is missing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for metric models. Use 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )

    parser.add_argument("--alignscore_ckpt", type=str, default="")
    parser.add_argument("--alignscore_model", type=str, default="roberta-large")
    parser.add_argument("--alignscore_eval_mode", type=str, default="nli_sp")
    parser.add_argument("--alignscore_batch_size", type=int, default=32)
    parser.add_argument(
        "--no_alignscore_adamw_patch",
        dest="alignscore_patch_adamw",
        action="store_false",
        help="Disable compatibility patch for alignscore 0.1.x with newer transformers.",
    )
    parser.add_argument(
        "--no_alignscore_pkg_resources_patch",
        dest="alignscore_patch_pkg_resources",
        action="store_false",
        help="Disable compatibility patch for lightning_fabric without pkg_resources.",
    )
    parser.set_defaults(
        alignscore_patch_adamw=True,
        alignscore_patch_pkg_resources=True,
    )

    args = parser.parse_args()
    if not args.in_csv and not args.in_dir:
        parser.error("Supply --in_csv or --in_dir.")
    if args.in_csv and not args.out_csv:
        parser.error("Supply --out_csv with --in_csv.")
    if bool(args.in_dir) != bool(args.out_dir):
        parser.error("--in_dir and --out_dir must be provided together.")
    return args


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)
    jobs = collect_csv_jobs(args)
    if args.in_dir:
        print(f"Found {len(jobs)} CSV files in {args.in_dir}")

    scorer = None
    for index, (in_csv, out_csv) in enumerate(jobs, start=1):
        if len(jobs) > 1:
            print(f"\n[{index}/{len(jobs)}] Processing {in_csv.name}")
        scorer = evaluate_csv(in_csv, out_csv, args, scorer)

    if len(jobs) > 1:
        print(f"\n[Done] processed {len(jobs)} CSV files.")


if __name__ == "__main__":
    main()
