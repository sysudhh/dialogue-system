#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 INPUTFILE"
  exit 1
fi

inputFile="$1"
echo "$inputFile"

python test.py "$inputFile"