#!/bin/sh

export PYTHONPATH=.:$PYTHONPATH
cd /livedar
python3 ./dar.py $@
