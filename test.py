from dotenv import load_dotenv
from fincost.util.alpha_vantage import AlphaVantageWrapper

load_dotenv()

av_wrapper = AlphaVantageWrapper()
daily_news = av_wrapper.get_global_news("2025-06-15")
print(daily_news)