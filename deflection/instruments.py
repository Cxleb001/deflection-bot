"""Instrument definitions: currency conversion and fallback spreads.

Names follow OANDA style: BASE_QUOTE, e.g. EUR_USD, USD_JPY, XAU_USD.

The account currency is USD. P&L is earned in the QUOTE currency, so:
  * X_USD instruments (EUR_USD, XAU_USD, ...): P&L is already in USD.
  * USD_X instruments (USD_JPY, USD_CAD, ...): P&L is converted by
    dividing by the price.
  * Crosses (EUR_GBP, ...): need a third rate - not supported yet.

DEFAULT_SPREADS are rough, deliberately conservative PLACEHOLDERS used only
when your data has no real bid/ask spread column. Data downloaded with
`python -m deflection download` includes real historical spreads, which
always take priority. Replace these with your broker's real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SPREADS = {
    "EUR_USD": 0.00014,
    "GBP_USD": 0.00018,
    "AUD_USD": 0.00016,
    "NZD_USD": 0.00020,
    "USD_JPY": 0.016,
    "USD_CAD": 0.00020,
    "USD_CHF": 0.00018,
    "XAU_USD": 0.45,
    "XAG_USD": 0.04,
}


@dataclass(frozen=True)
class Instrument:
    name: str
    default_spread: float | None = None

    @property
    def base(self) -> str:
        return self.name.split("_")[0]

    @property
    def quote(self) -> str:
        return self.name.split("_")[1]

    def __post_init__(self):
        parts = self.name.split("_")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Instrument names look like EUR_USD or XAU_USD, got {self.name!r}")
        if self.quote != "USD" and self.base != "USD":
            raise ValueError(
                f"{self.name} is a cross (neither side is USD). Converting its P&L to USD "
                f"needs a third exchange rate, which isn't supported yet."
            )

    def to_usd(self, amount_in_quote: float, price: float) -> float:
        """Convert an amount in the quote currency to USD at `price`."""
        if self.quote == "USD":
            return amount_in_quote
        return amount_in_quote / price  # base is USD, e.g. USD_JPY: JPY -> USD

    def usd_notional_per_unit(self, price: float) -> float:
        """How many USD one unit of the instrument is worth (for leverage caps)."""
        if self.quote == "USD":
            return price          # 1 EUR (or 1 oz gold) costs `price` USD
        return 1.0                # 1 unit of USD_JPY is 1 USD


def get_instrument(name: str) -> Instrument:
    name = name.upper()
    return Instrument(name=name, default_spread=DEFAULT_SPREADS.get(name))
