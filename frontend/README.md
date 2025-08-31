# Live Volatility Dashboard

Lightweight static frontend for real-time (polling) display of last price and model forecasts.

## Serve

Option 1: Via FastAPI static mount (add later) or simply open `index.html` in a browser while API runs locally.

## API Endpoints Used

* `GET /live/forecast/{symbol}` – snapshot includes last price + forecasts.

## Symbols

Currently hard-coded: `RELIANCE`, `HDFCBANK`, `INFY`.

Adjust in `index.html` (SYMBOLS array) or extend backend `start_live_updater` call.

## Notes

* Refresh interval: 2 minutes.
* Intraday fetch interval: 120s (config in `live.py`).
* No incremental model updating yet; forecasts reuse loaded daily models.
