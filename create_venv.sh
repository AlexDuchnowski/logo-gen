#!/bin/bash

python3 -m venv logo_env
source ./logo_env/bin/activate
pip install -U pip
pip install -r requirements.txt

echo created environment