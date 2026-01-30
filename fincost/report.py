import json
import os
from collections import Counter

from .commission import format_action_summary


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
