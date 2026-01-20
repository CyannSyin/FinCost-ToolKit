from __future__ import annotations

from dotenv import load_dotenv

from fincost.backtest import run_backtest
from fincost.config import load_settings
from fincost.data import build_data
from fincost.io import build_csv_paths, init_csv_files
from fincost.llm import create_trading_agent, get_llm
from fincost.tools import build_tools


def main() -> None:
    load_dotenv()

    settings = load_settings()
    df, full_df, start_date_for_filename, end_date = build_data(settings)

    print("[Step 2] Initializing LLM...")
    llm = get_llm(settings)
    print("[Step 2] LLM initialized successfully.")

    tools = build_tools(df)

    print("[Step 3] Creating agent...")
    agent = create_trading_agent(llm, tools=tools)
    print("[Step 3] Agent created successfully.")

    paths = build_csv_paths(settings, start_date_for_filename)
    init_csv_files(paths)

    run_backtest(settings, df, full_df, agent, paths, end_date)


if __name__ == "__main__":
    main()
