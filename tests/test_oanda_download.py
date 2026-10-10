"""OANDA downloader, tested against fake API responses (no network, no token)."""

import pandas as pd
import pytest

import deflection.data as data


def candle(time, bid_open, spread=0.0002, complete=True):
    b, a = bid_open, bid_open + spread
    fmt = lambda x: f"{x:.5f}"
    return {
        "complete": complete,
        "volume": 42,
        "time": time,
        "bid": {"o": fmt(b), "h": fmt(b + 0.0010), "l": fmt(b - 0.0010), "c": fmt(b + 0.0002)},
        "ask": {"o": fmt(a), "h": fmt(a + 0.0010), "l": fmt(a - 0.0010), "c": fmt(a + 0.0002)},
    }


def ts(i):
    return (pd.Timestamp("2024-01-02T00:00:00Z") + pd.Timedelta(minutes=15 * i)).strftime(
        "%Y-%m-%dT%H:%M:%S.000000000Z"
    )


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": dict(params)})
        return self.responses.pop(0)


def test_candles_to_frame_builds_mid_prices_and_spread():
    df = data.candles_to_frame([candle(ts(0), 1.10000), candle(ts(1), 1.10100, complete=False)])
    assert len(df) == 1                                    # incomplete candle dropped
    row = df.iloc[0]
    assert row["open"] == pytest.approx(1.10010)            # (bid + ask) / 2
    assert row["spread"] == pytest.approx(0.0002)
    assert str(df.index.tz) == "UTC"


def test_download_paginates_without_duplicates(monkeypatch):
    monkeypatch.setattr(data._time, "sleep", lambda s: None)
    pages = [
        FakeResponse(200, {"candles": [candle(ts(i), 1.1 + i * 1e-4) for i in (0, 1, 2)]}),
        FakeResponse(200, {"candles": [candle(ts(i), 1.1 + i * 1e-4) for i in (3, 4, 5)]}),
        FakeResponse(200, {"candles": [candle(ts(6), 1.1006), candle(ts(7), 1.1007, complete=False)]}),
    ]
    session = FakeSession(pages)
    df = data.download_oanda("EUR_USD", "M15", "2024-01-02", token="TESTTOKEN",
                             session=session, page_size=3)

    assert len(df) == 7
    assert df.index.is_unique and df.index.is_monotonic_increasing
    first, second = session.calls[0], session.calls[1]
    assert first["url"] == "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
    assert first["headers"]["Authorization"] == "Bearer TESTTOKEN"
    assert first["params"]["price"] == "BA"
    assert first["params"]["includeFirst"] == "true"
    assert second["params"]["includeFirst"] == "false"
    assert second["params"]["from"].startswith("2024-01-02T00:30:00")   # resumes from last candle


def test_download_retries_after_rate_limit(monkeypatch):
    monkeypatch.setattr(data._time, "sleep", lambda s: None)
    session = FakeSession([
        FakeResponse(429, {"errorMessage": "slow down"}),
        FakeResponse(200, {"candles": [candle(ts(0), 1.1)]}),
    ])
    df = data.download_oanda("EUR_USD", "M15", "2024-01-02", token="T", session=session)
    assert len(df) == 1 and len(session.calls) == 2


def test_download_requires_a_token(monkeypatch):
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="OANDA_API_TOKEN"):
        data.download_oanda("EUR_USD", "M15", "2024-01-02", session=FakeSession([]))


def test_csv_round_trip(tmp_path):
    df = data.synthetic_ohlc(n=50, seed=1)
    path = tmp_path / "x.csv"
    data.save_csv(df, path)
    back = data.load_csv(path)
    assert len(back) == 50
    assert back["close"].to_numpy() == pytest.approx(df["close"].to_numpy())
    assert "spread" in back.columns
