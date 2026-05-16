import argparse
from pathlib import Path

import pandas as pd

from ml_training.common import find_column


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


def _validate_dataset(path: Path, text_candidates, label_candidates, dataset_name: str) -> list[str]:
    issues: list[str] = []

    if not path.exists():
        issues.append(f"Missing file: {path}")
        return issues

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        issues.append(f"Cannot read CSV {path}: {exc}")
        return issues

    if df.empty:
        issues.append(f"{dataset_name}: dataset is empty")
        return issues

    label_col = find_column(df, label_candidates)
    if not label_col:
        issues.append(
            f"{dataset_name}: no label column found. Expected one of {label_candidates}"
        )

    text_col = find_column(df, text_candidates)

    # URL datasets can be either raw-url based or pre-engineered feature tables.
    if dataset_name == "URL dataset":
        if not text_col:
            non_label_columns = [col for col in df.columns if str(col).lower() != str(label_col).lower()] if label_col else list(df.columns)
            if len(non_label_columns) < 2:
                issues.append(
                    f"{dataset_name}: needs either a URL column {text_candidates} or precomputed feature columns"
                )
    else:
        if not text_col:
            issues.append(
                f"{dataset_name}: no text/url column found. Expected one of {text_candidates}"
            )

    if label_col and df[label_col].nunique(dropna=True) < 2:
        issues.append(f"{dataset_name}: label column '{label_col}' has fewer than 2 classes")

    return issues


def validate_all_datasets(datasets_dir: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []

    issues.extend(
        _validate_dataset(
            datasets_dir / "url_phishing.csv",
            ["url", "domain", "link"],
            ["label", "result", "class", "target", "status"],
            "URL dataset",
        )
    )

    issues.extend(
        _validate_dataset(
            datasets_dir / "email_phishing.csv",
            ["text", "email", "content", "message", "body"],
            ["label", "class", "target", "result"],
            "Email dataset",
        )
    )

    sms_path = datasets_dir / "sms_spam.csv"
    sms_issues = _validate_dataset(
        sms_path,
        ["text", "sms", "message", "content"],
        ["label", "class", "target", "result"],
        "SMS dataset",
    )

    # Support the classic UCI SMS Spam file stored in sms_spam.csv: tab-separated, no header.
    if sms_issues:
        try:
            sms_df = pd.read_csv(sms_path, sep="\t", names=["label", "text"], header=None)
            if sms_df.empty:
                sms_issues = ["SMS dataset: dataset is empty"]
            elif sms_df["label"].nunique(dropna=True) < 2:
                sms_issues = ["SMS dataset: label column has fewer than 2 classes"]
            else:
                sms_issues = []
        except Exception:
            pass

    issues.extend(sms_issues)

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset files and column requirements")
    parser.add_argument("--datasets-dir", type=str, default=str(DATASETS_DIR))
    args = parser.parse_args()

    ok, issues = validate_all_datasets(Path(args.datasets_dir))

    if ok:
        print("[SUCCESS] All datasets are valid and ready for training.")
        return 0

    print("[ERROR] Dataset validation failed:")
    for issue in issues:
        print(f" - {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
