import os
import requests
import datetime


class AlphaVantageWrapper:
    """
    Alpha Vantage API Wrapper.
    source: https://www.alphavantage.co/documentation/
    """

    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.base_url = f"https://www.alphavantage.co/query"

    def get_global_news(self, date: str) -> str:
        """
        Get global news articles for the specified date.
        Alpha Vantage Market News & Sentiment
        """

        if not self.api_key:
            return "Missing ALPHA_VANTAGE_API_KEY"

        # Alpha Vantage expects YYYYMMDDTHHMM format (e.g., 20250803T0000).
        day_start = datetime.datetime.strptime(date, "%Y-%m-%d")
        time_from = day_start.strftime("%Y%m%dT%H%M")
        time_to = day_start.replace(hour=23, minute=59).strftime("%Y%m%dT%H%M")
        try:
            response = requests.get(self.base_url, params={
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets",
                "time_from": time_from,
                "time_to": time_to,
                "limit": 10,
                "apikey": self.api_key,
        })
            data = response.json()
            results = []
            for item in data.get("feed", []) or []:
                if not isinstance(item, dict):
                    continue
                summary = item.get("summary") or ""
                if len(summary) > 600:
                    summary = summary[:597] + "..."
                results.append(
                    {
                        "title": item.get("title"),
                        "time_published": item.get("time_published"),
                        "source": item.get("source"),
                        "url": item.get("url"),
                        "summary": summary,
                        "overall_sentiment_label": item.get("overall_sentiment_label"),
                        "overall_sentiment_score": item.get("overall_sentiment_score"),
                    }
                )
            return results
        except Exception as exc:
            print(f"Error retrieving global news: {exc}")
            return "Error retrieving global news"
