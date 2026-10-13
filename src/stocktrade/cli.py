"""Command-line research workflows. No live order submission exists."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stocktrade", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a SYNTHETIC software test, not a historical backtest")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--frequency-seconds", type=int, default=60)
    training = sub.add_parser("train", help="Fit on chronological train data; hold out validation/test")
    training.add_argument("--bars", required=True)
    training.add_argument("--events")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--lookback", type=int, default=16)
    training.add_argument("--train-fraction", type=float, default=.7)
    training.add_argument("--validation-fraction", type=float, default=.15)
    evaluation = sub.add_parser("evaluate", help="Estimate observed execution analogs; unsupported orders abstain")
    evaluation.add_argument("--bars", required=True)
    evaluation.add_argument("--events")
    evaluation.add_argument("--trades", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--cash", type=float, default=100000)
    news = sub.add_parser("collect-news", help="Fetch a permitted HTTPS RSS/Atom feed; record observation time")
    news.add_argument("--url", required=True)
    news.add_argument("--output", required=True)
    news.add_argument("--symbol")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-news":
            from .data import collect_rss
            result = dict(collected_events=collect_rss(args.url, args.output, symbol=args.symbol))
        else:
            from .pipeline import evaluate, train
            if args.command == "demo":
                from .demo import generate_demo
                output = Path(args.output)
                data = generate_demo(output / "synthetic", frequency_seconds=args.frequency_seconds, seed=args.seed)
                train(data / "bars.csv", data / "events.jsonl", output / "model", epochs=args.epochs, seed=args.seed)
                result = evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
                                  output / "model/model.pt", output / "evaluation")
                result["demo_warning"] = "SYNTHETIC software validation only, not investment performance"
            elif args.command == "train":
                result = train(args.bars, args.events, args.output, epochs=args.epochs, seed=args.seed,
                               lookback=args.lookback, train_fraction=args.train_fraction,
                               validation_fraction=args.validation_fraction)
            else:
                result = evaluate(args.bars, args.events, args.trades, args.checkpoint, args.output, cash=args.cash)
        console = dict(result)
        if "execution" in console:
            console["execution"] = {key: value for key, value in result["execution"].items()
                                    if key not in {"records", "equity_curve", "source_trade_ids"}}
            console["report_path"] = str(Path(args.output) / ("evaluation/report.json"
                                            if args.command == "demo" else "report.json"))
        print(json.dumps(console, indent=2, default=str, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"stocktrade: {exc}", file=sys.stderr)
        return 1
    return 0
