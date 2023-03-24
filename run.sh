#!/bin/bash
rm -rf /tmp/tpu_logs
bash ./clear_tpu_locks.sh
# python3 ./test.py
python3 ./run.py