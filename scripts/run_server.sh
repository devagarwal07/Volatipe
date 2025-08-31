#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
