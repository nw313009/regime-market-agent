import os, requests
from dotenv import load_dotenv
load_dotenv()

key = os.environ["MASSIVE_API_KEY"]

# ── VERIFY BOTH OF THESE against Massive's live docs before running ──
BASE = "https://api.massive.com"
AGGS = f"{BASE}/v2/aggs/ticker/NVDA/range/1/day/2026-07-01/2026-08-01"
NEWS = f"{BASE}/v2/reference/news"   # check the real news route + its params

print("── aggregates ──")
r = requests.get(AGGS, params={"apiKey": key}, timeout=30)
print(r.status_code)
print(r.text[:800])

print("\n── news ──")
r2 = requests.get(NEWS, params={"apiKey": key, "ticker": "NVDA", "limit": 5}, timeout=30)
print(r2.status_code)
print(r2.text[:800])