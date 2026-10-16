"""Command-line action-policy research workflows; no live order submission exists."""

import argparse
from dataclasses import fields
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


def _risk_config(path):
    if path is None:
        return None
    from .risk import RiskConfig
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Risk configuration must be a JSON object containing RiskConfig field overrides")
    unknown = set(value) - {field.name for field in fields(RiskConfig)}
    if unknown:
        raise ValueError(f"Unknown risk configuration fields: {', '.join(sorted(unknown))}")
    if any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in value.values()):
        raise ValueError("Risk configuration values must be numeric; lot_size must be a positive integer")
    return RiskConfig(**value)


def _console_report(result, args):
    console = dict(result)
    if "execution" in console:
        console["execution"] = {
            key: value for key, value in result["execution"].items()
            if key not in {"records", "equity_curve", "source_trade_ids"}
        }
    if "diagnostics" in console:
        console["diagnostics"] = {key: result["diagnostics"][key]
                                  for key in ("evaluation_only", "used_for_training", "decisions")}
    if args.command in {"evaluate", "demo", "report"}:
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
    demo.add_argument("--episode-steps", type=int, default=128)
    demo.add_argument("--risk-config", help="JSON file containing hard risk-limit overrides")
    training = sub.add_parser("train", help="Train a policy to maximize episode terminal portfolio wealth")
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
    training.add_argument("--risk-config", help="JSON file containing hard risk-limit overrides")
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
    reporting = sub.add_parser("report", help="Visualize archived decisions without running or training a policy")
    reporting.add_argument("--bars", required=True)
    reporting.add_argument("--report", required=True, help="Archived evaluation report.json")
    reporting.add_argument("--decisions", required=True, help="Archived decisions.jsonl")
    reporting.add_argument("--output", required=True, help="New output directory preserving the source audit")
    for reporter in (demo, evaluation, reporting):
        reporter.add_argument("--no-dashboard", action="store_true", help="Write JSON/CSV diagnostics without HTML")
        reporter.add_argument("--benchmark", help="Observed reference stock in the supplied bars (evaluation only)")
        reporter.add_argument("--risk-free-rate", type=float, default=0.0, help="Annual reference rate, evaluation only")
        reporter.add_argument("--risk-lookback", type=int, default=60, help="Past return window for decision diagnostics")
        reporter.add_argument("--risk-confidence", type=float, default=.95, help="Historical VaR/ES confidence")
        reporter.add_argument("--risk-min-observations", type=int, default=20)
    action = sub.add_parser("act", help="Propose risk-checked dollar trading actions from current information")
    action.add_argument("--bars", required=True)
    action.add_argument("--events")
    action.add_argument("--checkpoint", required=True)
    action.add_argument("--funding", type=float,
                        help="Total initial funding; defaults to the checkpoint's configured funding")
    action.add_argument("--symbols", nargs="+", help="Restrict trading to a subset of checkpoint stocks")
    action.add_argument("--holdings", help="JSON file mapping allowed stock symbols to whole-share quantities")
    action.add_argument("--cash", type=float, help="Current uninvested cash; required when shares are held")
    action.add_argument("--peak-equity", type=float, help="Actual historical peak equity for drawdown controls")
    action.add_argument("--day-start-equity", type=float, help="Actual equity at today's start for daily-loss controls")
    action.add_argument("--offline", action="store_true", help="Use supplied events without online collection")
    action.add_argument("--sources", help="Online information source configuration JSON file")
    action.add_argument("--collection-output", default="data/online-events.jsonl",
                        help="Append online observations with their actual availability timestamps")
    action.add_argument("--max-market-age-seconds", type=float, default=900.0,
                        help="Maximum market-data age for online action proposals")
    action.add_argument("--output", help="Optional JSON file for proposed actions; no order is submitted")
    collection = sub.add_parser("collect-information", help="Gather configured online information for stocks")
    collection.add_argument("--symbols", nargs="+", required=True)
    collection.add_argument("--output", required=True)
    collection.add_argument("--sources", help="Online information source configuration JSON file")
    alpaca_bars = sub.add_parser("fetch-alpaca-bars", help="Import read-only Alpaca minute bars with provenance")
    alpaca_news = sub.add_parser("fetch-alpaca-news", help="Import read-only Alpaca news at actual retrieval time")
    for importer in (alpaca_bars, alpaca_news):
        importer.add_argument("--symbols", nargs="+", required=True)
        importer.add_argument("--start", required=True, help="ISO 8601 timestamp including a UTC offset")
        importer.add_argument("--end", required=True, help="ISO 8601 timestamp including a UTC offset")
        importer.add_argument("--timeout", type=float, default=15.0)
    alpaca_bars.add_argument("--output", required=True, help="New directory for bars.csv and provenance.json")
    alpaca_bars.add_argument("--feed", choices=("iex", "sip"), default="iex")
    alpaca_news.add_argument("--output", required=True, help="New JSONL file for archived news events")
    news = sub.add_parser("collect-news", help="Fetch a permitted HTTPS RSS/Atom feed; record observation time")
    news.add_argument("--url", required=True)
    news.add_argument("--output", required=True)
    news.add_argument("--symbol")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-news":
            from .data import collect_rss
            result = dict(collected_events=collect_rss(args.url, args.output, symbol=args.symbol))
        elif args.command == "collect-information":
            from .collection import collect_information
            result = collect_information(tuple(args.symbols), args.output, config_path=args.sources).to_dict()
        elif args.command == "fetch-alpaca-bars":
            from .platforms import fetch_alpaca_bars
            result = fetch_alpaca_bars(tuple(args.symbols), args.start, args.end, args.output,
                                      feed=args.feed, timeout=args.timeout)
        elif args.command == "fetch-alpaca-news":
            from .platforms import fetch_alpaca_news
            result = fetch_alpaca_news(tuple(args.symbols), args.start, args.end, args.output, timeout=args.timeout)
        elif args.command == "report":
            from .diagnostics import EvaluationConfig, regenerate_report
            config = EvaluationConfig(risk_lookback=args.risk_lookback, risk_confidence=args.risk_confidence,
                risk_min_observations=args.risk_min_observations, benchmark_symbol=args.benchmark,
                risk_free_rate=args.risk_free_rate)
            result = regenerate_report(args.bars, args.report, args.decisions, args.output,
                                       config=config, dashboard=not args.no_dashboard)
        else:
            from .pipeline import act, evaluate, train, write_json
            symbols = tuple(args.symbols) if args.symbols else None
            if args.command == "demo":
                from .diagnostics import EvaluationConfig
                config = EvaluationConfig(risk_lookback=args.risk_lookback, risk_confidence=args.risk_confidence,
                    risk_min_observations=args.risk_min_observations, benchmark_symbol=args.benchmark,
                    risk_free_rate=args.risk_free_rate)
                from .demo import generate_demo
                output = Path(args.output)
                data = generate_demo(output / "synthetic", frequency_seconds=args.frequency_seconds,
                                     seed=args.seed, symbols=symbols)
                train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", output / "model",
                      epochs=args.epochs, seed=args.seed, funding=args.funding, symbols=symbols,
                      episode_steps=args.episode_steps,
                      risk_config=_risk_config(args.risk_config))
                result = evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
                                  output / "model/model.pt", output / "evaluation",
                                  funding=args.funding, symbols=symbols, evaluation_config=config,
                                  dashboard=not args.no_dashboard)
                result["demo_warning"] = "SYNTHETIC software validation only, not investment performance"
            elif args.command == "train":
                result = train(args.bars, args.events, args.trades, args.output, epochs=args.epochs,
                               seed=args.seed, lookback=args.lookback, funding=args.funding,
                               symbols=symbols,
                               risk_config=_risk_config(args.risk_config), episode_steps=args.episode_steps,
                               train_fraction=args.train_fraction, validation_fraction=args.validation_fraction)
            elif args.command == "evaluate":
                from .diagnostics import EvaluationConfig
                config = EvaluationConfig(risk_lookback=args.risk_lookback, risk_confidence=args.risk_confidence,
                    risk_min_observations=args.risk_min_observations, benchmark_symbol=args.benchmark,
                    risk_free_rate=args.risk_free_rate)
                result = evaluate(args.bars, args.events, args.trades, args.checkpoint, args.output,
                                  funding=args.funding, symbols=symbols, evaluation_config=config,
                                  dashboard=not args.no_dashboard)
            else:
                holdings = _holdings(args.holdings)
                if holdings and any(holdings.values()) and args.cash is None:
                    raise ValueError("--cash is required when --holdings contains invested shares")
                result = act(args.bars, args.events, args.checkpoint, funding=args.funding,
                             symbols=symbols, cash=args.cash, holdings=holdings,
                             peak_equity=args.peak_equity, day_start_equity=args.day_start_equity,
                             collect=not args.offline,
                             sources_path=args.sources, collection_output=args.collection_output,
                             max_market_age_seconds=args.max_market_age_seconds)
                if args.output:
                    write_json(args.output, result)
        print(json.dumps(_console_report(result, args), indent=2, default=str, allow_nan=False))
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"stocktrade: {exc}", file=sys.stderr)
        return 1
    return 0
