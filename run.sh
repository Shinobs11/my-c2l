#!/bin/bash
rm -rf /tmp/tpu_logs
PJRT_DEVICE=TPU MASTER_ADDR=localhost MASTER_PORT=6000 python3 ./run.py