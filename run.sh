#!/bin/bash
# export XLA_SAVE_HLO_FILE="/tmp/xla.log"
# export XLA_IR_DEBUG=true
# export XLA_HLO_DEBUG=true
# export XLA_DUMP_HLO_GRAPH=true
rm -rf /tmp/tpu_logs
bash ./clear_tpu_locks.sh

# python3 ./test.py


killall python3 -9
# XLA_DUMP_HLO_GRAPH=true \
# XLA_SAVE_HLO_FILE="/tmp/xla.log" \
# XLA_IR_DEBUG=true \
# XLA_HLO_DEBUG=true \
# PT_XLA_DEBUG=1 python3 ./run.py

python3 ./run.py