import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import get_project_root, load_app_config, load_static_config


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return records


def _relaxed_json_loads(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(cleaned)


def _load_static_any(path: str) -> dict[str, Any]:
    raw = _read_text(path).strip()
    if not raw:
        return {}

    try:
        data = _relaxed_json_loads(raw)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if "structure" in data:
            try:
                return load_static_config(path)
            except (json.JSONDecodeError, ValueError):
                return data
        return data

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_count": len(records),
        "date_range": None,
        "models": [],
        "decision_counts": {},
        "trade_count": 0,
        "hold_ratio": None,
        "token_total": 0,
        "token_average": None,
        "latency_average_ms": None,
        "tool_counts": {},
        "tool_latency_total_ms": 0,
        "slippage_avg": None,
        "slippage_pct_avg": None,
        "analysis_to_decision_latency_ms": None,
    }
    if not records:
        return summary

    dates = [str(r.get("date")) for r in records if r.get("date")]
    if dates:
        summary["date_range"] = (min(dates), max(dates))

    models = {r.get("model") or r.get("signature") for r in records if r.get("model") or r.get("signature")}
    summary["models"] = sorted(models)

    decision_counter = Counter()
    token_total = 0
    latencies = []
    tool_counter = Counter()
    tool_latency_total = 0
    slippages = []
    slippage_pcts = []
    decision_latencies = []
    trade_total = 0

    for record in records:
        llm_usage = record.get("llm_usage") or {}
        token_total += int(llm_usage.get("input_tokens", 0)) + int(llm_usage.get("output_tokens", 0))
        latency_ms = llm_usage.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            latencies.append(float(latency_ms))

        tool_usage = record.get("tool_usage") or {}
        for call in tool_usage.get("calls", []) or []:
            tool_name = call.get("tool_name")
            if tool_name:
                tool_counter[tool_name] += int(call.get("call_count", 0) or 0)
                tool_latency_total += int(call.get("total_latency_ms", 0) or 0)

        trades = record.get("trades") or []
        trade_total += len(trades)
        for trade in trades:
            decision_type = trade.get("decision_type") or "UNKNOWN"
            decision_counter[decision_type] += 1
            analysis_price = trade.get("analysis_price")
            execution_price = trade.get("execution_price")
            if isinstance(analysis_price, (int, float)) and isinstance(execution_price, (int, float)):
                slippage = float(execution_price) - float(analysis_price)
                slippages.append(slippage)
                if analysis_price:
                    slippage_pcts.append(slippage / float(analysis_price))
            timestamp = trade.get("timestamp") or {}
            analysis_ts = _parse_ts(timestamp.get("analysis_time"))
            decision_ts = _parse_ts(timestamp.get("decision_time"))
            if analysis_ts and decision_ts:
                decision_latencies.append((decision_ts - analysis_ts).total_seconds() * 1000)

    summary["decision_counts"] = dict(decision_counter)
    summary["trade_count"] = trade_total
    hold_count = decision_counter.get("HOLD", 0)
    if trade_total:
        summary["hold_ratio"] = hold_count / trade_total

    summary["token_total"] = token_total
    if records:
        summary["token_average"] = token_total / len(records)
    if latencies:
        summary["latency_average_ms"] = sum(latencies) / len(latencies)
    summary["tool_counts"] = dict(tool_counter)
    summary["tool_latency_total_ms"] = tool_latency_total
    if slippages:
        summary["slippage_avg"] = sum(slippages) / len(slippages)
    if slippage_pcts:
        summary["slippage_pct_avg"] = sum(slippage_pcts) / len(slippage_pcts)
    if decision_latencies:
        summary["analysis_to_decision_latency_ms"] = sum(decision_latencies) / len(decision_latencies)

    return summary


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 200]
    return head + "\n\n[...truncated...]\n"


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"record_count: {summary.get('record_count')}",
        f"date_range: {summary.get('date_range')}",
        f"models: {summary.get('models')}",
        f"decision_counts: {summary.get('decision_counts')}",
        f"trade_count: {summary.get('trade_count')}",
        f"hold_ratio: {summary.get('hold_ratio')}",
        f"token_total: {summary.get('token_total')}",
        f"token_average: {summary.get('token_average')}",
        f"latency_average_ms: {summary.get('latency_average_ms')}",
        f"tool_counts: {summary.get('tool_counts')}",
        f"tool_latency_total_ms: {summary.get('tool_latency_total_ms')}",
        f"slippage_avg: {summary.get('slippage_avg')}",
        f"slippage_pct_avg: {summary.get('slippage_pct_avg')}",
        f"analysis_to_decision_latency_ms: {summary.get('analysis_to_decision_latency_ms')}",
    ]
    return "\n".join(lines)


def _resolve_llm(model: str) -> ChatOpenAI:
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=model,
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )

    aihubmix_key = os.getenv("AIHUBMIX_API_KEY")
    if aihubmix_key:
        base_url = os.getenv("AIHUBMIX_BASE_URL")
        if not base_url:
            raise ValueError("AIHUBMIX_BASE_URL is required when using AIHUBMIX_API_KEY")
        return ChatOpenAI(
            model=model,
            api_key=aihubmix_key,
            base_url=base_url,
            temperature=0.2,
        )

    raise ValueError("Missing API key: set OPENAI_API_KEY or AIHUBMIX_API_KEY in env")


def _default_output_path(md_path: str) -> str:
    md_dir = os.path.dirname(os.path.abspath(md_path))
    base = os.path.basename(md_path)
    if base.endswith(".md"):
        base = base[:-3]
    filename = f"{base}-diagnosis.md"
    return os.path.join(md_dir, filename)


def run_diagnosis(md_path: str, records_path: str, static_path: str, output_path: str | None = None) -> str:
    root = get_project_root()
    load_dotenv(os.path.join(root, ".env"))
    app_config = load_app_config(os.path.join(root, "config.json"))
    model = str(app_config.get("agent_model", "gpt-4o-mini"))

    md_text = _read_text(md_path)
    records = _load_jsonl(records_path)
    static_config = _load_static_any(static_path)

    summary = _summarize_records(records)
    static_preview = json.dumps(static_config, ensure_ascii=False, indent=2)

    prompt = "\n".join(
        [
            "You are a FinCost analysis agent.",
            "Use the provided report, experiment records summary, and static config to:",
            "1) Diagnose model performance (strengths, weaknesses, anomalies).",
            "2) Explain likely causes with evidence from the data.",
            "3) Provide concrete improvement suggestions (prompting, tools, trading logic, risk controls, latency, cost).",
            "4) List quick wins vs long-term improvements.",
            "",
            "## Report (markdown excerpt)",
            _truncate(md_text, 6000),
            "",
            "## Experiment Records (summary)",
            _format_summary(summary),
            "",
            "## Static Config",
            _truncate(static_preview, 4000),
            "",
            "Return a concise markdown report with clear sections and bullet points.",
        ]
    )

    llm = _resolve_llm(model)
    response = llm.invoke(
        [
            SystemMessage(content="You are a meticulous trading-system performance diagnostician."),
            HumanMessage(content=prompt),
        ]
    )
    diagnosis = response.content.strip()

    output_path = output_path or _default_output_path(md_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(diagnosis + "\n")

    return diagnosis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinCost model diagnosis agent")
    parser.add_argument("--md", required=True, help="Path to report markdown file")
    parser.add_argument("--records", required=True, help="Path to experiment_records JSONL")
    parser.add_argument("--static", required=True, help="Path to static config JSON/JSONL")
    parser.add_argument("--output", help="Optional path to save diagnosis markdown")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    diagnosis = run_diagnosis(args.md, args.records, args.static, args.output)
    print(diagnosis)


if __name__ == "__main__":
    main()
