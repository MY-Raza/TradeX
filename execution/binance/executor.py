from binance.client import Client
from dotenv import load_dotenv
import os

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..', '.env'))
load_dotenv(dotenv_path)

API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")

client = Client(API_KEY, API_SECRET)

# VERY IMPORTANT → Switch to Futures Testnet
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

# Test connection
account_info = client.futures_account()
print(account_info)
