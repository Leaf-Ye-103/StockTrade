"""Command-line action-policy research workflows; no live order submission exists."""

import argparse
import json
import sys
from pathlib import Path


def _holdings(path):
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Holdings must be a JSON object mapping stock symbols to whole-share quantities")
    if any(not isinstance(symbol, str) or not symbol.strip()
           or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0
           for symbol, quantity in value.items()):
        raise ValueError("Holdings require nonempty stock symbols and nonnegative whole-share quantities")
    return value


def _console_report(result, args):
    console = dict(result)
    if "execution" in console:
        console["execution"] = {
            key: value for key, value in result["execution"].items()
            if key not in {"records", "equity_curve", "source_trade_ids"}
        }
    if args.command in {"evaluate", "demo"}:
        directory = Path(args.output) / "evaluation" if args.command == "demo" else Path(args.output)
        console["report_path"] = str(directory / "report.json")
    elif args.command == "act" and args.output:
        console["report_path"] = str(args.output)
    return console


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stocktrade", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a SYNTHETIC software test, not a historical backtest")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--frequency-seconds", type=int, default=60)
    demo.add_argument("--funding", type=float, default=100000.0)
    demo.add_argument("--symbols", nargs="+", default=["NVDA", "MSFT", "AAPL"])
    demo.add_argument("--reward-horizon", type=int, default=8)
    training = sub.add_parser("train", help="Train an action policy from chronological observed-trade feedback")
    training.add_argument("--bars", required=True)
    training.add_argument("--events")
    training.add_argument("--trades", required=True,
                          help="Recorded executions used to evaluate actions; unsupported fills are rejected")
    training.add_argument("--output", required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--lookback", type=int, default=16)
    training.add_argument("--funding", type=float, default=100000.0)
    training.add_argument("--symbols", nargs="+", help="Allowed stock symbols; defaults to all input stocks")
    training.add_argument("--reward-horizon", type=int, default=8)
    training.add_argument("--selection-drawdown-weight", type=float, default=.5,
                          help="Validation checkpoint score penalizes portfolio drawdown by this weight")
    training.add_argument("--constraint-penalty", type=float, default=.001)
    training.add_argument("--episode-steps", type=int, default=128)
    training.add_argument("--train-fraction", type=float, default=.7)
    training.add_argument("--validation-fraction", type=float, default=.15)
    evaluation = sub.add_parser("evaluate", help="Evaluate held-out actions using recorded execution evidence")
    evaluation.add_argument("--bars", required=True)
    evaluation.add_argument("--events")
    evaluation.add_argument("--trades", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--funding", "--cash", dest="funding", type=float,
                            help="Starting funding; defaults to the checkpoint's configured funding")
    evaluation.add_argument("--symbols", nargs="+", help="Restrict trading to a subset of checkpoint stocks")
    action = sub.add_parser("act", help="Propose risk-checked dollar trading actions from current information")
    action.add_argument("--bars", required=True)
    action.add_argument("--events")
    action.add_argument("--checkpoint", required=True)
    action.add_argument("--funding", type=float,
                        help="Total initial funding; defaults to the checkpoint's configured funding")
    action.add_argument("--symbols", nargs="+", help="Restrict trading to a subset of checkpoint stocks")
    action.add_argument("--holdings", help="JSON file mapping allowed stock symbols to whole-share quantities")
    action.add_argument("--cash", type=float, help="Current uninvested cash; required when shares are held")
    action.add_argument("--output", help="Optional JSON file for proposed actions; no order is submitted")
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
            from .pipeline import act, evaluate, train, write_json
            symbols = tuple(args.symbols) if args.symbols else None
            if args.command == "demo":
                from .demo import generate_demo
                output = Path(args.output)
                data = generate_demo(output / "synthetic", frequency_seconds=args.frequency_seconds,
                                     seed=args.seed, symbols=symbols)
                train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", output / "model",
                      epochs=args.epochs, seed=args.seed, funding=args.funding, symbols=symbols,
                      reward_horizon=args.reward_horizon)
                result = evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
                                  output / "model/model.pt", output / "evaluation",
                                  funding=args.funding, symbols=symbols)
                result["demo_warning"] = "SYNTHETIC software validation only, not investment performance"
            elif args.command == "train":
                result = train(args.bars, args.events, args.trades, args.output, epochs=args.epochs,
                               seed=args.seed, lookback=args.lookback, funding=args.funding,
                               symbols=symbols, reward_horizon=args.reward_horizon,
                               selection_drawdown_weight=args.selection_drawdown_weight,
                               constraint_penalty=args.constraint_penalty, episode_steps=args.episode_steps,
                               train_fraction=args.train_fraction, validation_fraction=args.validation_fraction)
            elif args.command == "evaluate":
                result = evaluate(args.bars, args.events, args.trades, args.checkpoint, args.output,
                                  funding=args.funding, symbols=symbols)
            else:
                holdings = _holdings(args.holdings)
                if holdings and any(holdings.values()) and args.cash is None:
                    raise ValueError("--cash is required when --holdings contains invested shares")
                result = act(args.bars, args.events, args.checkpoint, funding=args.funding,
                             symbols=symbols, cash=args.cash, holdings=holdings)
                if args.output:
                    write_json(args.output, result)
        print(json.dumps(_console_report(result, args), indent=2, default=str, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"stocktrade: {exc}", file=sys.stderr)
        return 1
    return 0
