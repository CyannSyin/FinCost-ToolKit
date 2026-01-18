from __future__ import annotations

import pandas as pd
from langchain.tools import tool


def build_tools(df: pd.DataFrame):
    @tool
    def get_current_price(date: str) -> float:
        """Get TSLA closing price (prices) for the specified date."""
        row = df[df["start"] == date]
        if len(row) == 0:
            return None
        return float(row.iloc[0]["prices"])

    @tool
    def get_historical_prices(days: int = 10) -> str:
        """Get closing price sequence for the last 'days' days (for Agent to reference trends)."""
        if len(df) < days:
            return str(df["prices"].tolist())
        prices = df["prices"].tail(days).tolist()
        return str(prices)

    @tool
    def get_news(date: str) -> str:
        """Get TSLA news articles for the specified date."""
        row = df[df["start"] == date]
        if len(row) == 0:
            return f"No data found for date: {date}"

        try:
            news_list = row.iloc[0]["news"]

            if news_list is None:
                return f"No news available for date: {date}"

            if isinstance(news_list, list):
                if len(news_list) == 0:
                    return f"No news available for date: {date}"
                if len(news_list) > 0 and isinstance(news_list[0], str):
                    return "\n\n" + "=" * 80 + "\n\n".join(
                        [
                            f"News Article {i+1}:\n{article}"
                            for i, article in enumerate(news_list)
                        ]
                    )
                return str(news_list)
            if isinstance(news_list, str):
                return news_list
            return str(news_list)
        except Exception as exc:
            return f"Error retrieving news for date {date}: {str(exc)}"

    return [get_current_price, get_historical_prices, get_news]
