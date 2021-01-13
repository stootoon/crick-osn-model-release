#!/bin/bash
# Runs classify on all combinations of conc, delay and shuffle settings
echo "Classifying CONCENTRATION."
python -u classify.py "$@"
echo "Classifying CONCENTRATION (SHUFFLED)."
python -u classify.py "$@" --shuf
echo "Classifying CONCENTRATION & DELAY."
python -u classify.py "$@" --delay
echo "Classifying CONCENTRATION & DELAY (SHUFFLED)."
python -u classify.py "$@" --delay --shuf

