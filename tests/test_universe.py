"""Universe tests — LCW filtering (mocked HTTP) + availability split."""
import compression_detection.universe as u


class FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


def test_filter_available_split():
    available = {"ADAUSDT", "BTCUSDT", "HYPEUSDT"}
    avail, miss = u.filter_available(["ADA/USDT", "XYZ/USDT", "HYPE/USDT"], available)
    assert avail == ["ADA/USDT", "HYPE/USDT"]
    assert miss == ["XYZ/USDT"]


def test_filter_available_none_passes_everything():
    avail, miss = u.filter_available(["ADA/USDT", "XYZ/USDT"], None)
    assert avail == ["ADA/USDT", "XYZ/USDT"]
    assert miss == []


def test_pair_to_binance_symbol():
    assert u.pair_to_binance_symbol("ada/usdt") == "ADAUSDT"
    assert u.pair_to_binance_symbol("1000SATS/USDT") == "1000SATSUSDT"


def test_fetch_coins_filters(monkeypatch):
    data = [
        {"code": "btc", "volume": 1000, "categories": [], "rate": 1},
        {"code": "ADA", "volume": 500, "categories": [], "rate": 1},
        {"code": "USDC", "volume": 900, "categories": ["stablecoins"], "rate": 1},
        {"code": "ada", "volume": 100, "categories": [], "rate": 1},
        {"code": "LOWVOL", "volume": 10, "categories": [], "rate": 1},
    ]
    monkeypatch.setattr(u.requests, "post", lambda *a, **k: FakeResp(data))
    coins = u.fetch_coins("key", limit=300, min_volume_btc=100, btc_price=2.0)
    # BTC kept (user rule 2026-09-21), stablecoin dropped, dup keeps the
    # higher volume, threshold=200
    assert [c["symbol"] for c in coins] == ["BTC", "ADA"]


def test_get_universe_pairs(monkeypatch):
    data = [
        {"code": "ETH", "volume": 500, "categories": [], "rate": 1},
    ]
    monkeypatch.setattr(u.requests, "post", lambda *a, **k: FakeResp(data))
    pairs = u.get_universe("key", min_volume_btc=100, btc_price=2.0)
    assert pairs == ["ETH/USDT"]
