import os
import random

from .analysis import (
    calculate_cumulative_cost_series,
    calculate_monthly_cost_series,
    calculate_portfolio_state,
    calculate_portfolio_series,
    calculate_token_cost_by_date,
    calculate_opportunity_cost_and_latency,
)
from .commission import extract_actions_and_commission
from .config import get_project_root, load_app_config, load_llm_pricing, load_static_config
from .diagnosis_render import render_diagnosis_to_html
from .plots import plot_cost_pie, plot_performance_lines
from .render import render_markdown_to_html
from .agent import run_diagnosis
from .records import load_experiment_records, load_daily_buy_prices
from .report import (
    build_report_payload,
    build_report_text,
    build_summary_bill_markdown,
    save_report_jsonl,
    save_report_markdown,
    save_report_text,
)


def main():
    root = get_project_root()
    data_dir = os.path.join(root, "data")
    app_config = load_app_config(os.path.join(root, "config.json"))

    records_path = app_config.get(
        "records_path",
        os.path.join(
            data_dir,
            "experiment_records_gpt-4o-mini_10000.0_2026-01-06-14-00-00_2026-01-07-14-00-00.jsonl",
        ),
    )
    static_path = app_config.get(
        "static_path",
        os.path.join(data_dir, "static_gpt-4o-mini_10000.jsonl"),
    )
    prices_path = app_config.get("prices_path", os.path.join(data_dir, "merged.jsonl"))

    records = load_experiment_records(records_path)
    prices_by_date = load_daily_buy_prices(prices_path)
    model_pricing = load_llm_pricing(os.path.join(root, "config_llm.json"))
    static_config = load_static_config(static_path)

    actions_by_date, commission_by_date, commission_total = extract_actions_and_commission(records)

    trading_days = len(actions_by_date)
    _, token_cost_by_date = calculate_token_cost_by_date(records, model_pricing)
    token_total = sum(token_cost_by_date.values())
    infra_cost_per_day = 0.2

    monthly_cost = float(static_config.get("data_subscription_monthly", 0.0))
    monthly_additions, monthly_total = calculate_monthly_cost_series(records, monthly_cost)
    uncertain_cost = sum(random.uniform(0.0, 0.5) for _ in range(trading_days))
    infra_total = trading_days * infra_cost_per_day
    total_cost = commission_total + token_total + infra_total + monthly_total + uncertain_cost

    report_text = build_report_text(
        actions_by_date,
        trading_days,
        commission_total,
        token_total,
        infra_total,
        monthly_total,
        uncertain_cost,
        total_cost,
    )
    print(report_text)

    llm_model = str(static_config.get("llm_model", "unknown"))
    initial_cash = str(static_config.get("initial_cash", "unknown"))
    frequency = str(static_config.get("decision_frequency", static_config.get("frequency", "unknown")))

    result_dir = os.path.join(root, "result", f"{llm_model}-{initial_cash}-{frequency}")
    os.makedirs(result_dir, exist_ok=True)

    save_report_text(report_text, result_dir, llm_model, initial_cash, frequency)
    portfolio_state = calculate_portfolio_state(records, initial_cash, prices_by_date)
    opportunity_cost, average_latency_ms, trade_count = calculate_opportunity_cost_and_latency(
        records
    )
    bill_markdown = build_summary_bill_markdown(
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
    )
    bill_markdown_path = save_report_markdown(
        bill_markdown, result_dir, llm_model, initial_cash, frequency
    )
    render_markdown_to_html(bill_markdown_path)
    try:
        run_diagnosis(bill_markdown_path, records_path, static_path)
        diagnosis_md_path = bill_markdown_path
        if diagnosis_md_path.endswith(".md"):
            diagnosis_md_path = f"{diagnosis_md_path[:-3]}-diagnosis.md"
        render_diagnosis_to_html(diagnosis_md_path)
    except Exception as exc:
        print(f"\n[Warning] Failed to run diagnosis agent: {exc}")
    report_payload = build_report_payload(
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
    )
    save_report_jsonl(report_payload, result_dir, llm_model, initial_cash, frequency)

    pie_dir = os.path.join(result_dir, "pie-chart")
    os.makedirs(pie_dir, exist_ok=True)
    pie_filename = f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_pie_chart.pdf"
    pie_path = os.path.join(pie_dir, pie_filename)
    plot_cost_pie(commission_total, token_total, infra_total, monthly_total, uncertain_cost, pie_path)
    pie_filename_no_monthly = (
        f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_pie_chart_no_monthly.pdf"
    )
    pie_path_no_monthly = os.path.join(pie_dir, pie_filename_no_monthly)
    plot_cost_pie(
        commission_total,
        token_total,
        infra_total,
        0.0,
        uncertain_cost,
        pie_path_no_monthly,
        include_monthly=False,
    )

    dates, holding_profit_series = calculate_portfolio_series(
        records, initial_cash, prices_by_date
    )
    uncertain_additions = [uncertain_cost] + [0.0 for _ in records[1:]]
    combined_additions = [
        monthly_additions[i] + uncertain_additions[i] for i in range(len(monthly_additions))
    ]
    cumulative_cost_series = calculate_cumulative_cost_series(
        records,
        commission_by_date,
        token_cost_by_date,
        infra_cost_per_day,
        combined_additions,
    )
    real_profit_series = [
        holding_profit_series[i] - cumulative_cost_series[i]
        for i in range(len(holding_profit_series))
    ]
    line_dir = os.path.join(result_dir, "line chart")
    os.makedirs(line_dir, exist_ok=True)
    line_filename = f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_line_chart.pdf"
    line_path = os.path.join(line_dir, line_filename)
    plot_performance_lines(
        dates,
        holding_profit_series,
        cumulative_cost_series,
        real_profit_series,
        line_path,
    )

    no_monthly_additions = [0.0 for _ in records]
    combined_additions_no_monthly = [
        no_monthly_additions[i] + uncertain_additions[i]
        for i in range(len(no_monthly_additions))
    ]
    cumulative_cost_series_no_monthly = calculate_cumulative_cost_series(
        records,
        commission_by_date,
        token_cost_by_date,
        infra_cost_per_day,
        combined_additions_no_monthly,
    )
    real_profit_series_no_monthly = [
        holding_profit_series[i] - cumulative_cost_series_no_monthly[i]
        for i in range(len(holding_profit_series))
    ]
    line_filename_no_monthly = (
        f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_line_chart_no_monthly.pdf"
    )
    line_path_no_monthly = os.path.join(line_dir, line_filename_no_monthly)
    plot_performance_lines(
        dates,
        holding_profit_series,
        cumulative_cost_series_no_monthly,
        real_profit_series_no_monthly,
        line_path_no_monthly,
    )


if __name__ == "__main__":
    main()
