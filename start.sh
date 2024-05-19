#!/bin/bash
pip install -r requirements.txt
nohup uvicorn server.main:app --reload --host=0.0.0.0 --port=10010 > server.log 2>&1 &