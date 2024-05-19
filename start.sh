#!/bin/bash
pip install -r requirements.txt
uvicorn server.main:app --reload --host=0.0.0.0 --port=10010