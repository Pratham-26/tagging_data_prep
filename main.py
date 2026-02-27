import argparse

from src.gepa_pipeline.config import load_config
from src.gepa_pipeline.optimizer import run_optimization

from src.loader import load_hierarchy
from src.modules.hierarchical_classifier import HierarchicalClassifier


def cmd_optimize(args):
    """Run GEPA optimization to generate optimized prompts."""
    print(f"Loading config from {args.config}...")
    result = run_optimization(args.config)
    print(f"Optimized {len(result)} paths")
    for path_key in result:
        print(f"  {path_key}")


def cmd_classify(args):
    """Classify text using optimized prompts."""
    config = load_config(args.config)
    hierarchy = load_hierarchy(config.paths.hierarchy)
    classifier = HierarchicalClassifier(hierarchy=hierarchy, prompts_path=config.paths.prompts)

    if args.text:
        results = classifier.classify_batch([args.text])
        for r in results:
            print(f"Path: {' > '.join(r.path)}")
            print(f"Status: {r.status.value}")
    elif args.input:
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
        results = classifier.classify_batch(texts)
        for r in results:
            path_str = " > ".join(r.path) if r.path else "(none)"
            print(f"{path_str}\t{r.status.value}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical text classifier with GEPA optimization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    opt_parser = subparsers.add_parser("optimize", help="Run GEPA optimization")
    opt_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    opt_parser.set_defaults(func=cmd_optimize)

    cls_parser = subparsers.add_parser("classify", help="Classify text")
    cls_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    cls_parser.add_argument("--text", help="Text to classify")
    cls_parser.add_argument("--input", help="File with texts to classify (one per line)")
    cls_parser.set_defaults(func=cmd_classify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
