from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from shadow_validation.audit import audit_run  # noqa: E402
from shadow_validation.report import write_audit_artifacts  # noqa: E402


SUPPORTED_INPUT_SUFFIXES = frozenset({".csv", ".xlsx", ".xlsm"})


def label_files(label_dir: str | Path) -> list[Path]:
    """Return label files in a stable order and reject an empty inbox clearly."""
    root = Path(label_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"라벨 입력 폴더가 없습니다: {root}")
    files = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and not path.name.startswith("~$")
            and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not files:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"라벨 입력 폴더에 지원 파일이 없습니다 ({supported}): {root}")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit one sealed MES shadow prediction run")
    parser.add_argument("--run-dir", required=True, help="sealed prediction run directory")
    parser.add_argument(
        "--label-dir",
        default=PACKAGE_ROOT / "label_inbox",
        help="directory containing CSV/XLSX/XLSM label files",
    )
    parser.add_argument(
        "--output-dir",
        help="directory for the static audit artifacts (default: <run-dir>/audit)",
    )
    parser.add_argument("--final", action="store_true", help="require every shadow row to have a label")
    parser.add_argument(
        "--as-of-date",
        help="YYYY-MM-DD cutoff; only fully labelled months before its calendar month count as complete",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "audit"
    try:
        if not run_dir.is_dir():
            raise ValueError(f"봉인된 예측 실행 폴더가 없습니다: {run_dir}")
        result = audit_run(
            run_dir,
            label_files(args.label_dir),
            final=args.final,
            as_of_date=args.as_of_date,
        )
        paths = write_audit_artifacts(result, output_dir)
    except Exception as error:
        print(f"감사 실패: {error}", file=sys.stderr)
        return 2
    print(f"감사 결과: {paths['report.html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
