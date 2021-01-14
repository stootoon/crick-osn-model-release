#!/bin/bash
# Runs classify on all combinations of conc, delay and shuffle settings
echo "Classifying CONCENTRATION."
python -u classify.py "$@"
echo "Classifying CONCENTRATION (SHUFFLED)."
python -u classify.py "$@" --shuf
echo "Classifying CONCENTRATION & PPI."
python -u classify.py "$@" --ppi
echo "Classifying CONCENTRATION & PPI (SHUFFLED)."
python -u classify.py "$@" --ppi --shuf

