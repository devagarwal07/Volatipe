#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/{raw,processed,features,predictions}
mkdir -p models logs results mlruns
