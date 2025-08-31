#!/bin/bash
source venv/bin/activate
python -m src.ingestion.nse_data --start-date 2018-01-01 --end-date $(date +%Y-%m-%d)
python -m src.ingestion.vix_data --start-date 2018-01-01 --end-date $(date +%Y-%m-%d)
