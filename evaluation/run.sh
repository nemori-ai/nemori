#!/bin/bash

# Make sure you have set OPENAI_API_KEY in your environment
# export OPENAI_API_KEY=your_api_key_here
python locomo/add.py
python locomo/search.py
python locomo/evals.py
python locomo/generate_scores.py