#!/bin/bash

for i in $(seq 1 10)
do
  PYTHONPATH="./:${PYTHONPATH}" python scripts/metrics/table_calculate_fid_allx2_set14.py
done

