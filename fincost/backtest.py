from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import Settings
from .indicators import technical_agent


def calculate_commission(
    shares: int,
    trade_value: float,
    commission_per_share: float,
    commission_minimum: float,
    commission_maximum_rate: float,
) -> float:
    if shares == 0 or trade_value == 0:
        return 0.0

    base_commission = shares * commission_per_share
    commission = max(base_commission, commission_minimum)
    max_commission = trade_value * commission_maximum_rate
    commission = min(commission, max_commission)
    return commission


def run_backtest(
    settings: Settings,
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    agent,
    paths: Dict[str, Path],
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    initial_cash = float(settings.initial_cash)
    max_days = len(df) - 1

    print(f"[Step 4] Starting backtest with {max_days} days...")
    print(f"[Step 4] Initial cash: ${initial_cash:.2f}")

    main_csv_path = paths["main_csv_path"]
    llm_outputs_csv_path = paths["llm_outputs_csv_path"]
    llm_inputs_csv_path = paths["llm_inputs_csv_path"]

    print("[Step 4] Results will be saved to:")
    print(f"  - Main results: {main_csv_path}")
    print(f"  - LLM outputs: {llm_outputs_csv_path}")
    print(f"  - LLM inputs: {llm_inputs_csv_path}")

    records = []
    llm_outputs_records = []

    cash = initial_cash
    shares = 0
    cumulative_cost = 0.0
    cumulative_gross_profit = 0.0

    current_month = None

    for i in range(max_days):
        if i % 10 == 0:
            print(f"[Progress] Day {i+1}/{max_days}")
        current_row = df.iloc[i]
        date_str = current_row["start"]
        current_price = float(current_row["prices"])

        current_date = pd.to_datetime(date_str)
        current_date_month = (current_date.year, current_date.month)

        monthly_subscription_cost_today = 0.0
        if current_month is None:
            current_month = current_date_month
            monthly_subscription_cost_today = settings.data_subscription_monthly
        elif current_date_month != current_month:
            current_month = current_date_month
            monthly_subscription_cost_today = settings.data_subscription_monthly

        history_prices = df.iloc[max(0, i - 9) : i + 1]["prices"].tolist()
        signal_results = technical_agent(full_df, date_str)

        try:
            if "news" not in df.columns:
                news_content = "News column not available in dataset."
            else:
                current_news = df.iloc[i]["news"]

                if pd.isna(current_news) or current_news is None:
                    news_content = "No news available for this date."
                elif isinstance(current_news, list):
                    if len(current_news) > 0:
                        if isinstance(current_news[0], str):
                            valid_articles = [
                                article
                                for article in current_news
                                if article and isinstance(article, str) and article.strip()
                            ]
                            if valid_articles:
                                news_content = "\n\n".join(
                                    [
                                        f"News Article {idx+1}:\n{article}"
                                        for idx, article in enumerate(valid_articles)
                                    ]
                                )
                            else:
                                news_content = "No news available for this date."
                        else:
                            news_content = "\n\n".join(
                                [
                                    f"News Article {idx+1}:\n{str(article)}"
                                    for idx, article in enumerate(current_news)
                                    if article
                                ]
                            )
                            if not news_content:
                                news_content = "No news available for this date."
                    else:
                        news_content = "No news available for this date."
                elif isinstance(current_news, str):
                    if current_news.strip():
                        news_content = current_news
                    else:
                        news_content = "No news available for this date."
                else:
                    news_str = str(current_news)
                    if news_str and news_str.strip() and news_str.lower() not in [
                        "nan",
                        "none",
                        "null",
                    ]:
                        news_content = news_str
                    else:
                        news_content = "No news available for this date."
        except KeyError:
            news_content = "News column not available in dataset."
        except Exception as exc:
            news_content = f"Error retrieving news: {str(exc)}"

        recent_news_list = []
        if "news" in df.columns:
            for j in range(max(0, i - 2), i + 1):
                try:
                    day_news = df.iloc[j]["news"]
                    day_date = df.iloc[j]["start"]

                    if pd.isna(day_news) or day_news is None:
                        continue
                    if isinstance(day_news, list) and len(day_news) > 0:
                        valid_articles = [
                            article
                            for article in day_news[:2]
                            if article and isinstance(article, str) and article.strip()
                        ]
                        if valid_articles:
                            formatted_news = "\n".join(
                                [
                                    f"  - {article[:200]}..."
                                    if len(article) > 200
                                    else f"  - {article}"
                                    for article in valid_articles
                                ]
                            )
                            recent_news_list.append(
                                f"Date {day_date}:\n{formatted_news}"
                            )
                    elif isinstance(day_news, str) and day_news.strip():
                        formatted_news = (
                            f"  - {day_news[:200]}..."
                            if len(day_news) > 200
                            else f"  - {day_news}"
                        )
                        recent_news_list.append(f"Date {day_date}:\n{formatted_news}")
                except Exception:
                    pass

        recent_news_summary = (
            "\n\n".join(recent_news_list)
            if recent_news_list
            else "No recent news available."
        )

        try:
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(
                        content="""
You are a trading agent analyzing market data to make investment decisions about TSLA stock.
Based on your analysis, provide a specific recommendation to buy, sell, or hold. 

IMPORTANT: You will receive the current trading date in each request. Pay attention to the date as it helps you understand the temporal context of the market data.

MANDATORY REQUIREMENT: You MUST analyze the news information provided in each request. News articles contain critical information about market sentiment, company developments, regulatory changes, earnings, product launches, and other factors that significantly impact stock prices. Your decision MUST be based on BOTH price data AND news analysis. Ignoring news information will lead to poor trading decisions.

AVAILABLE TOOLS (optional, news is already provided in the prompt):
You have access to the following tools if you need additional information:
1. get_current_price(date): Get TSLA closing price for a specific date
2. get_historical_prices(days): Get closing price sequence for the last N days
3. get_news(date): Get TSLA news articles for a specific date (use if you need news from other dates)

Do not forget to utilize lessons from past decisions to learn from your mistakes. 
Here are some reflections from similar situations you traded in and the lessons learned:
- Consider market trends and volatility patterns
- Balance risk and reward based on available capital
- Avoid emotional trading decisions
- Learn from previous gains and losses
- ALWAYS analyze news alongside price data - news often provides context for price movements

IMPORTANT: You must provide your decision reasoning BEFORE stating your decision.

Output format:
1. First, explain your analysis and reasoning for the decision (decision rationale)
   - You MUST explain how the news information influenced your decision
   - Analyze both positive and negative news signals
   - Consider how news relates to price trends
2. Then, conclude with your decision in the exact format:
   - For buy: "buy <number>" where <number> is the number of shares to buy (positive integer)
   - For sell: "sell <number>" where <number> is the number of shares to sell (positive integer)
   - For hold: "hold" (no number needed)

Examples:
- "Based on the price trend showing a 10% increase over the last week, positive news about production expansion indicating strong demand, and current momentum, I recommend buying. The news suggests continued growth potential. buy 100"
- "The stock price has been declining for 5 consecutive days with high volatility, and recent news indicates regulatory concerns and potential fines. This combination of negative price action and negative news suggests potential further decline. sell 50"
- "The price is relatively stable with no clear trend, and recent news is neutral with no significant developments. I want to wait for more signals before making a move. hold"

Always provide your reasoning before the decision statement.
"""
                    ),
                    HumanMessage(
                        content=f"""Trading Date: {date_str}

=== PRICE DATA ===
Last 10 days closing price sequence: {history_prices}
Today's closing price: ${current_price:.3f}

=== PORTFOLIO STATUS ===
Available cash: ${cash:.2f}
Current shares: {shares}

=== TECHNICAL SIGNALS ===
{json.dumps(signal_results, ensure_ascii=False)}

=== NEWS INFORMATION (MANDATORY TO ANALYZE) ===
Today's News ({date_str}):
{news_content}

Recent News Summary (Last 3 days):
{recent_news_summary}

=== INSTRUCTIONS ===
Please analyze BOTH the price data AND the news information provided above. You MUST consider how the news affects market sentiment and stock price movements. Based on your comprehensive analysis of price trends and news content, provide your investment decision in the format: buy <number>, sell <number>, or hold:"""
                    ),
                ]
                if i == 0:
                    print(f"[API] Calling agent for day {i+1} (date: {date_str})...")
                result = agent.invoke({"messages": messages})
                if i == 0:
                    print("[API] Agent response received.")

                decision_raw = ""
                llm_output_raw = ""

                if result.get("messages"):
                    last_ai_message = None
                    for msg in reversed(result["messages"]):
                        if isinstance(msg, AIMessage):
                            last_ai_message = msg
                            break

                    if last_ai_message:
                        llm_output_raw = (
                            last_ai_message.content if last_ai_message.content else ""
                        )
                        decision_raw = llm_output_raw.strip().lower()
                    else:
                        last_message = result["messages"][-1]
                        if hasattr(last_message, "content"):
                            llm_output_raw = (
                                last_message.content
                                if last_message.content
                                else str(last_message)
                            )
                        else:
                            llm_output_raw = str(last_message)
                        decision_raw = llm_output_raw.strip().lower()
                else:
                    llm_output_raw = str(result)
                    decision_raw = llm_output_raw.strip().lower()

                input_tokens = cb.prompt_tokens
                output_tokens = cb.completion_tokens
                llm_cost = (
                    input_tokens * settings.input_price_per_k_tokens
                    + output_tokens * settings.output_price_per_k_tokens
                )

                input_content_parts = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        input_content_parts.append(f"[SYSTEM MESSAGE]\n{msg.content}")
                    elif isinstance(msg, HumanMessage):
                        input_content_parts.append(f"[HUMAN MESSAGE]\n{msg.content}")
                input_content = "\n\n".join(input_content_parts)

                llm_input_record = {
                    "date": date_str,
                    "input_content": input_content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

                pd.DataFrame([llm_input_record]).to_csv(
                    llm_inputs_csv_path, mode="a", header=False, index=False
                )
        except Exception as exc:
            print(
                f"[ERROR] Failed to get agent decision at day {i+1} (date: {date_str}): {exc}"
            )
            raise

        action = "hold"
        quantity = 0

        buy_match = re.search(r"\bbuy\s+(\d+)\b", decision_raw)
        sell_match = re.search(r"\bsell\s+(\d+)\b", decision_raw)

        if buy_match:
            quantity = int(buy_match.group(1))
            if quantity > 0:
                action = "buy"
            else:
                action = "hold"
                quantity = 0
        elif sell_match:
            quantity = int(sell_match.group(1))
            if quantity > 0:
                action = "sell"
            else:
                action = "hold"
                quantity = 0
        elif re.search(r"\bhold\b", decision_raw) or not decision_raw.strip():
            action = "hold"
            quantity = 0
        else:
            action = "hold"
            quantity = 0
            if "buy" in decision_raw.lower() or "sell" in decision_raw.lower():
                print(
                    f"Warning: Unrecognized format at {date_str}. Expected 'buy <number>', "
                    f"'sell <number>', or 'hold'. Got: {decision_raw[:100]}"
                )

        if i + 1 >= len(df):
            break
        exec_price = float(df.iloc[i + 1]["prices"])

        trade_count = 0
        trade_value = 0.0
        slippage = 0.0
        actual_quantity = 0

        if action == "buy" and quantity > 0:
            max_affordable = int(cash / exec_price)
            actual_quantity = min(quantity, max_affordable)

            if actual_quantity > 0:
                trade_value = exec_price * actual_quantity
                shares += actual_quantity
                cash -= trade_value
                trade_count = 1
                slippage = (
                    (exec_price - current_price) * actual_quantity
                    if exec_price > current_price
                    else 0
                )

        elif action == "sell" and quantity > 0:
            actual_quantity = min(quantity, shares)

            if actual_quantity > 0:
                trade_value = exec_price * actual_quantity
                shares -= actual_quantity
                cash += trade_value
                trade_count = 1
                slippage = (
                    (current_price - exec_price) * actual_quantity
                    if exec_price < current_price
                    else 0
                )

        if trade_count > 0 and actual_quantity > 0:
            commission = calculate_commission(
                actual_quantity,
                trade_value,
                settings.commission_per_share,
                settings.commission_minimum,
                settings.commission_maximum_rate,
            )
        else:
            commission = 0.0
        trading_cost_today = commission

        infra_cost = settings.infra_daily
        random_cost = random.uniform(*settings.random_cost_range)
        daily_total_cost = (
            trading_cost_today
            + llm_cost
            + infra_cost
            + random_cost
            + monthly_subscription_cost_today
        )

        cumulative_cost += daily_total_cost

        market_value = cash + shares * exec_price
        cumulative_gross_profit = market_value - initial_cash
        cumulative_net_profit = cumulative_gross_profit - cumulative_cost

        record = {
            "date": date_str,
            "hold_shares": shares,
            "decision": action,
            "requested_quantity": quantity,
            "actual_quantity": actual_quantity,
            "market_value": round(market_value, 2),
            "available_cash": round(cash, 2),
            "daily_gross_pnl": round(
                cumulative_gross_profit
                - (
                    records[-1]["market_value"] + records[-1]["daily_gross_pnl"]
                    if records
                    else initial_cash
                ),
                2,
            ),
            "trade_count": trade_count,
            "daily_trading_cost": round(commission, 4),
            "daily_llm_input_tokens": input_tokens,
            "daily_llm_output_tokens": output_tokens,
            "daily_llm_cost_usd": round(llm_cost, 6),
            "daily_infra_cost": infra_cost,
            "daily_random_cost": round(random_cost, 4),
            "monthly_data_subscription_cost": round(monthly_subscription_cost_today, 4),
            "daily_total_cost": round(daily_total_cost, 4),
            "cumulative_cost": round(cumulative_cost, 2),
            "cumulative_net_profit": round(cumulative_net_profit, 2),
            "slippage_usd": round(slippage, 2),
        }
        records.append(record)

        pd.DataFrame([record]).to_csv(main_csv_path, mode="a", header=False, index=False)

        llm_record = {
            "date": date_str,
            "day": i + 1,
            "current_price": round(current_price, 3),
            "available_cash": round(cash, 2),
            "current_shares": shares,
            "decision": action,
            "requested_quantity": quantity,
            "actual_quantity": actual_quantity,
            "llm_output": llm_output_raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_cost_usd": round(llm_cost, 6),
        }
        llm_outputs_records.append(llm_record)
        pd.DataFrame([llm_record]).to_csv(
            llm_outputs_csv_path, mode="a", header=False, index=False
        )

        if end_date is not None:
            if current_date >= end_date:
                print(
                    f"[End Date Reached] Date: {date_str}, Reached specified end date: "
                    f"{settings.end_date}, Ran for {i+1} days"
                )
                break
        else:
            if market_value < daily_total_cost:
                print(
                    f"[Bankruptcy Termination] Date: {date_str}, Remaining market value: "
                    f"${market_value:.2f}, Ran for {i+1} days"
                )
                break

            recent_records = pd.DataFrame(records[-30:])
            avg_daily_cost = recent_records["daily_total_cost"].mean()
            avg_daily_net = recent_records["cumulative_net_profit"].diff().mean()
            if cumulative_net_profit >= 0 and avg_daily_net > avg_daily_cost * 1.1:
                print(
                    f"[Balance Achieved] Date: {date_str}, Time taken: {i+1} days, "
                    f"Cumulative net profit: ${cumulative_net_profit:.2f}"
                )
                break

    result_df = pd.DataFrame(records)

    print("\nBacktest completed! Results saved:")
    print(f"  - Main results: {main_csv_path}")
    print(f"  - LLM outputs: {llm_outputs_csv_path}")
    print(f"  - LLM inputs: {llm_inputs_csv_path}")
    print(f"Total trading days: {len(result_df)}")
    if len(result_df) > 0:
        print(f"Final market value: ${result_df.iloc[-1]['market_value']:.2f}")
        print(f"Cumulative total cost: ${result_df.iloc[-1]['cumulative_cost']:.2f}")
        print(
            f"Cumulative net profit: ${result_df.iloc[-1]['cumulative_net_profit']:.2f}"
        )

        print("\nLast 10 days records:")
        print(
            result_df[
                [
                    "date",
                    "hold_shares",
                    "decision",
                    "market_value",
                    "daily_total_cost",
                    "daily_llm_cost_usd",
                    "cumulative_cost",
                    "cumulative_net_profit",
                ]
            ].tail(10)
        )

    return result_df
