import json
import os
from collections import Counter

from .commission import format_action_summary


def _format_money(value):
    return f"{value:.2f}"


def build_report_text(
    actions_by_date,
    trading_days,
    commission_total,
    token_total,
    infra_total,
    monthly_total,
    uncertain_cost,
    total_cost,
):
    report_lines = [
        *format_action_summary(actions_by_date),
        "",
        "Cost totals:",
        f"  trading_days: {trading_days}",
        f"  commission_total: {commission_total:.2f}",
        f"  token_total: {token_total:.2f}",
        f"  infra_total: {infra_total:.2f}",
        f"  monthly_total: {monthly_total:.2f}",
        f"  uncertain_cost: {uncertain_cost:.2f}",
        f"  total_cost: {total_cost:.2f}",
    ]
    return "\n".join(report_lines)


def build_report_payload(
    llm_model,
    initial_cash,
    frequency,
    actions_by_date,
    trading_days,
    commission_total,
    token_total,
    infra_total,
    monthly_total,
    uncertain_cost,
    total_cost,
):
    return {
        "llm_model": llm_model,
        "initial_cash": initial_cash,
        "frequency": frequency,
        "action_summary_by_day": [
            {
                "date": date,
                "actions": dict(Counter(actions_by_date[date])),
            }
            for date in sorted(actions_by_date.keys())
        ],
        "cost_totals": {
            "trading_days": trading_days,
            "commission_total": round(commission_total, 2),
            "token_total": round(token_total, 2),
            "infra_total": round(infra_total, 2),
            "monthly_total": round(monthly_total, 2),
            "uncertain_cost": round(uncertain_cost, 2),
            "total_cost": round(total_cost, 2),
        },
    }


def save_report_text(report_text, result_dir, llm_model, initial_cash, frequency):
    txt_filename = f"{llm_model}-{initial_cash}-{frequency}.txt"
    txt_path = os.path.join(result_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nSaved report to: {txt_path}")


def save_report_jsonl(report_payload, result_dir, llm_model, initial_cash, frequency):
    jsonl_filename = f"{llm_model}_{initial_cash}_{frequency}.jsonl"
    result_path = os.path.join(result_dir, jsonl_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report_payload, ensure_ascii=False) + "\n")
    print(f"\nSaved report to: {result_path}")


def _resolve_trading_period(records, static_config):
    start_time = static_config.get("start_time")
    end_time = static_config.get("end_time")
    if start_time and end_time:
        return str(start_time), str(end_time)

    dates = [str(record.get("date") or "") for record in records if record.get("date")]
    if not dates:
        return "unknown", "unknown"
    return min(dates), max(dates)


def _resolve_strategy_identifier(records, static_config, llm_model):
    identifier = static_config.get("signature") or static_config.get("strategy_id")
    if identifier:
        return str(identifier)
    for record in records:
        signature = record.get("signature")
        if signature:
            return str(signature)
    return str(llm_model)


def build_summary_bill_markdown(
    records,
    static_config,
    llm_model,
    initial_cash,
    frequency,
    commission_total,
    token_total,
    infra_total,
    monthly_total,
    uncertain_cost,
    total_cost,
    portfolio_state,
    opportunity_cost,
    average_latency_ms,
    trade_count,
):
    start_time, end_time = _resolve_trading_period(records, static_config)
    strategy_id = _resolve_strategy_identifier(records, static_config, llm_model)

    assets = portfolio_state.get("assets", [])
    assets_text = ", ".join(assets) if assets else "N/A"

    positions = portfolio_state.get("positions", [])
    if positions:
        positions_lines = [
            f"- {item['ticker']}: {item['quantity']}"
            for item in positions
        ]
        positions_text = "\n".join(positions_lines)
    else:
        positions_text = "None"

    current_cash = portfolio_state.get("current_cash", 0.0)
    total_position_value = portfolio_state.get("total_position_value", 0.0)
    total_portfolio_value = portfolio_state.get("total_portfolio_value", 0.0)
    return_profit = portfolio_state.get("return_profit", 0.0)
    net_result = return_profit - float(total_cost)
    average_latency_text = (
        f"{average_latency_ms:.2f} ms" if trade_count else "N/A"
    )

    lines = [
        "# FinCost Summary Bill",
        "",
        "## 1. Trading Configuration",
        "",
        "Trading Period:",
        f"{start_time} - {end_time}",
        "",
        "Trading Model / Strategy:",
        strategy_id,
        "",
        "Assets:",
        assets_text,
        "",
        "## 2. Asset & Portfolio State",
        "",
        "Initial Cash:",
        _format_money(float(initial_cash)),
        "",
        "Current Cash:",
        _format_money(float(current_cash)),
        "",
        "Positions:",
        positions_text,
        "",
        "Total Position Value:",
        _format_money(float(total_position_value)),
        "",
        "Total Portfolio Value:",
        _format_money(float(total_portfolio_value)),
        "",
        "## 3. Performance",
        "",
        "Return / Profit:",
        _format_money(float(return_profit)),
        "",
        "## 4. Cost Breakdown",
        "### 4.1 Static Cost",
        "",
        "Costs that are independent of trading frequency and decision count.",
        "",
        "Data Subscription Cost",
        _format_money(float(monthly_total)),
        "",
        "### 4.2 Dynamic Cost",
        "",
        "Costs incurred as a result of system execution and decision-making.",
        "",
        "#### 4.2.1 Trading-side Cost",
        "",
        "Transaction Cost",
        _format_money(float(commission_total)),
        "",
        "#### 4.2.2 Model-side Cost",
        "",
        "Token Cost",
        _format_money(float(token_total)),
        "",
        "Infrastructure Cost",
        _format_money(float(infra_total)),
        "",
        "## 5. Uncertain Cost",
        "",
        "Costs that are economically relevant but not directly observable or precisely measurable.",
        "",
        "Uncertain Cost",
        _format_money(float(uncertain_cost)),
        "",
        "Opportunity Cost (Decision Price - Execution Price)",
        _format_money(float(opportunity_cost)),
        "",
        "Average Latency per Trade",
        average_latency_text,
        "",
        "## 6. Net Economic Outcome",
        "",
        "Total Cost:",
        _format_money(float(total_cost)),
        "",
        "Net Result:",
        _format_money(float(net_result)),
    ]
    return "\n".join(lines)


def save_report_markdown(report_markdown, result_dir, llm_model, initial_cash, frequency):
    md_filename = f"{llm_model}-{initial_cash}-{frequency}-bill.md"
    md_path = os.path.join(result_dir, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_markdown + "\n")
    print(f"\nSaved bill markdown to: {md_path}")
    return md_path
