from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Settings


def ensure_result_dir() -> Path:
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    return result_dir


def build_csv_paths(settings: Settings, start_date_for_filename: str) -> Dict[str, Path]:
    result_dir = ensure_result_dir()
    model_name_sanitized = (
        settings.llm_model.replace("/", "-").replace("\\", "-").replace(":", "-")
    )
    initial_cash_for_filename = (
        str(int(settings.initial_cash))
        if settings.initial_cash == int(settings.initial_cash)
        else str(settings.initial_cash).replace(".", "_")
    )

    main_csv_filename = (
        f"tsla_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
    )
    llm_outputs_csv_filename = (
        f"tsla_llm_outputs_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
    )
    llm_inputs_csv_filename = (
        f"tsla_llm_inputs_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
    )

    return {
        "main_csv_path": result_dir / main_csv_filename,
        "llm_outputs_csv_path": result_dir / llm_outputs_csv_filename,
        "llm_inputs_csv_path": result_dir / llm_inputs_csv_filename,
    }


def init_csv_files(paths: Dict[str, Path]) -> None:
    main_columns = [
        "date",
        "hold_shares",
        "decision",
        "requested_quantity",
        "actual_quantity",
        "market_value",
        "available_cash",
        "daily_gross_pnl",
        "trade_count",
        "daily_trading_cost",
        "daily_llm_input_tokens",
        "daily_llm_output_tokens",
        "daily_llm_cost_usd",
        "daily_infra_cost",
        "daily_random_cost",
        "monthly_data_subscription_cost",
        "daily_total_cost",
        "cumulative_cost",
        "cumulative_net_profit",
        "slippage_usd",
    ]
    llm_columns = [
        "date",
        "day",
        "current_price",
        "available_cash",
        "current_shares",
        "decision",
        "requested_quantity",
        "actual_quantity",
        "llm_output",
        "input_tokens",
        "output_tokens",
        "llm_cost_usd",
    ]
    llm_inputs_columns = ["date", "input_content", "input_tokens", "output_tokens", "total_tokens"]

    main_csv_path = paths["main_csv_path"]
    llm_outputs_csv_path = paths["llm_outputs_csv_path"]
    llm_inputs_csv_path = paths["llm_inputs_csv_path"]

    if not main_csv_path.exists():
        pd.DataFrame(columns=main_columns).to_csv(main_csv_path, index=False)
    if not llm_outputs_csv_path.exists():
        pd.DataFrame(columns=llm_columns).to_csv(llm_outputs_csv_path, index=False)
    if not llm_inputs_csv_path.exists():
        pd.DataFrame(columns=llm_inputs_columns).to_csv(llm_inputs_csv_path, index=False)
