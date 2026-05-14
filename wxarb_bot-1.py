"""
WXARB — Polymarket Weather Arbitrage Bot
Uses direct HTTP to Polymarket CLOB API + web3 for signing.
No py-clob-client dependency (fixes Railway build).
"""

import os, json, time, logging, requests, schedule, hashlib, hmac
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CAPITAL_USD       = float(os.getenv("CAPITAL_USD", 500))
MAX_BET_PCT       = float(os.getenv("MAX_BET_PCT", 0.05))
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", 0.30))
MIN_EDGE          = float(os.getenv("MIN_EDGE", 0.18))
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", 30))
DRY_RUN           = os.getenv("DRY_RUN", "true").lower() == "true"
ANTHROPIC_KEY     = os.getenv("ANTHROPIC_API_KEY")
POLY_PRIV_KEY     = os.getenv("POLYMARKET_PRIVATE_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wxarb")
client = Anthropic(api_key=ANTHROPIC_KEY)
session = {"trades": 0, "pnl": 0.0}

CLOB_HOST = "https://clob.polymarket.com"
CITIES = [
    {"name": "New York",    "lat": 40.71, "lon": -74.00},
    {"name": "Chicago",     "lat": 41.88, "lon": -87.63},
    {"name": "Los Angeles", "lat": 34.05, "lon": -118.24},
    {"name": "Miami",       "lat": 25.76, "lon": -80.19},
    {"name": "Dallas",      "lat": 32.78, "lon": -96.80},
]

# ── NOAA ──────────────────────────────────────────────────────────────────────
def get_noaa_forecast(city):
    try:
        r = requests.get(
            f"https://api.weather.gov/points/{city['lat']},{city['lon']}",
            headers={"User-Agent": "wxarb/1.0"}, timeout=10)
        r.raise_for_status()
        r2 = requests.get(r.json()["properties"]["forecast"],
            headers={"User-Agent": "wxarb/1.0"}, timeout=10)
        r2.raise_for_status()
        periods = r2.json()["properties"]["periods"]
        p = next((x for x in periods if "Tomorrow" in x.get("name", "") and x["isDaytime"]), periods[2])
        temp = p["temperature"]
        lo = (temp // 5) * 5
        return {
            "temp": temp,
            "range": f"{lo}-{lo+5}",
            "detail": p["shortForecast"],
            "confidence": 0.88 if any(w in p["shortForecast"].lower()
                for w in ["sunny", "clear", "mostly clear"]) else 0.78,
        }
    except Exception as e:
        log.warning(f"NOAA error {city['name']}: {e}")
        return None

# ── Polymarket CLOB (direct HTTP) ─────────────────────────────────────────────
def get_api_key():
    """Derive API key from private key via Polymarket auth endpoint."""
    try:
        from web3 import Web3
        from eth_account import Account
        w3 = Web3()
        acct = Account.from_key(POLY_PRIV_KEY)
        ts = str(int(time.time()))
        msg = f"This request will be used to authenticate with the CLOB API\nTimestamp: {ts}"
        signed = acct.sign_message(
            w3.eth.account._hash_eip191_message(
                w3.eth.account.encode_defunct(text=msg)
            )
        )
        r = requests.post(f"{CLOB_HOST}/auth/derive-api-key", json={
            "address": acct.address,
            "timestamp": ts,
            "signature": signed.signature.hex(),
        }, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Auth error: {e}")
        return None

def get_weather_markets():
    try:
        r = requests.get(f"{CLOB_HOST}/markets",
            params={"active": True, "tag": "weather"}, timeout=10)
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            q = m.get("question", "").lower()
            if "temperature" in q or "degrees" in q or "°f" in q:
                for t in m.get("tokens", []):
                    if t.get("outcome", "").lower() == "yes":
                        out.append({
                            "token_id": t["token_id"],
                            "question": m["question"],
                            "price": float(t.get("price", 0.5)),
                            "condition_id": m["condition_id"],
                        })
        return out
    except Exception as e:
        log.warning(f"Polymarket market fetch error: {e}")
        return []

def place_bet(token_id, amount_usd, price, api_creds):
    if DRY_RUN:
        log.info(f"[DRY RUN] ${amount_usd:.2f} on {token_id} @ {price:.2f}")
        return True
    try:
        size = round(amount_usd / price, 2)
        order = {
            "token_id": token_id,
            "price": price,
            "size": size,
            "side": "BUY",
            "order_type": "GTC",
        }
        r = requests.post(f"{CLOB_HOST}/order", json=order, headers={
            "POLY_ADDRESS": api_creds["address"],
            "POLY_SIGNATURE": api_creds["signature"],
            "POLY_TIMESTAMP": api_creds["timestamp"],
            "POLY_API_KEY": api_creds["apiKey"],
            "POLY_PASSPHRASE": api_creds["passphrase"],
        }, timeout=10)
        r.raise_for_status()
        log.info(f"Order placed: {r.json()}")
        return True
    except Exception as e:
        log.error(f"Order failed: {e}")
        return False

# ── Claude brain ──────────────────────────────────────────────────────────────
def ask_claude(city_name, noaa, market_price, capital):
    edge = noaa["confidence"] - market_price
    max_bet = capital * MAX_BET_PCT
    prompt = f"""You are a Polymarket weather arbitrage bot. Return ONLY valid JSON, no markdown.

City: {city_name}
NOAA: {noaa['temp']}°F, {noaa['detail']}
NOAA confidence for {noaa['range']}°F: {noaa['confidence']*100:.1f}%
Market price: {market_price*100:.1f}¢
Edge: {edge*100:.1f}%
Capital: ${capital:.2f} | Max bet: ${max_bet:.2f}

Kelly sizing: bet (edge / (1 - market_price)) * 0.25 of capital, capped at max_bet.
Be conservative.

Return: {{"shouldBet": true|false, "betSize": <dollars max {max_bet:.2f}>, "reasoning": "<one sentence>", "confidence": <0-1>}}"""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(msg.content[0].text.strip().replace("```json","").replace("```",""))
    except Exception as e:
        log.error(f"Claude error: {e}")
        return {"shouldBet": False, "betSize": 0, "reasoning": str(e), "confidence": 0}

# ── Main loop ─────────────────────────────────────────────────────────────────
def scan_and_trade():
    if session["pnl"] < -(CAPITAL_USD * STOP_LOSS_PCT):
        log.error(f"STOP LOSS HIT: ${session['pnl']:.2f}. Bot halted.")
        return schedule.CancelJob

    log.info(f"── SCAN | PnL: ${session['pnl']:+.2f} | Trades: {session['trades']} ──")
    capital = CAPITAL_USD + session["pnl"]

    api_creds = None
    if not DRY_RUN:
        api_creds = get_api_key()
        if not api_creds:
            log.error("Could not get API creds. Skipping cycle.")
            return

    markets = get_weather_markets()
    log.info(f"{len(markets)} weather markets found")

    for city in CITIES:
        noaa = get_noaa_forecast(city)
        if not noaa:
            continue
        log.info(f"{city['name']}: {noaa['temp']}°F | conf {noaa['confidence']*100:.0f}%")

        matched = [m for m in markets
                   if city["name"].lower() in m["question"].lower()
                   and noaa["range"] in m["question"]]
        if not matched:
            log.info(f"  No market found for {city['name']} {noaa['range']}°F")
            continue

        market = matched[0]
        edge = noaa["confidence"] - market["price"]
        log.info(f"  Market: {market['price']*100:.1f}¢ | Edge: {edge*100:+.1f}%")

        if edge < MIN_EDGE:
            log.info(f"  Edge {edge*100:.1f}% < min {MIN_EDGE*100:.0f}%. Skip.")
            continue

        decision = ask_claude(city["name"], noaa, market["price"], capital)
        log.info(f"  Claude: {decision['reasoning']}")

        if decision["shouldBet"] and decision["betSize"] > 0:
            if place_bet(market["token_id"], decision["betSize"], market["price"], api_creds):
                session["trades"] += 1

        time.sleep(2)

    log.info("── SCAN DONE ──")

if __name__ == "__main__":
    log.info(f"WXARB | ${CAPITAL_USD} | {'DRY RUN' if DRY_RUN else '⚡ LIVE'}")
    if not ANTHROPIC_KEY:
        log.error("Missing ANTHROPIC_API_KEY"); exit(1)
    if not DRY_RUN and not POLY_PRIV_KEY:
        log.error("Missing POLYMARKET_PRIVATE_KEY"); exit(1)

    scan_and_trade()
    schedule.every(SCAN_INTERVAL_MIN).minutes.do(scan_and_trade)
    log.info(f"Scanning every {SCAN_INTERVAL_MIN}min...")
    while True:
        schedule.run_pending()
        time.sleep(30)
