import argparse
import sys

from mflux.models.common.checking.model_checker import ModelChecker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check that a model package can run with MFlux.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    model_parser = subparsers.add_parser("model", help="Check a local or cached Hugging Face model package.")
    model_parser.add_argument("--fix", action="store_true", help="Try local repair without downloading files.")
    model_parser.add_argument("model_reference", help="A local path or an org/model Hugging Face reference.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = ModelChecker.check(args.model_reference)
    repair = None
    if args.fix and result.check == "warn" and result.format == "mflux":
        repair = ModelChecker.repair(args.model_reference)
        result = ModelChecker.check(args.model_reference)
    print(ModelChecker.format_result(result, args.model_reference, fix=args.fix, repair=repair))
    return 0 if result.check == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
