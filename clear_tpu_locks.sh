#!/bin/bash
#god i hope this works
shopt -s nullglob
for n in /dev/accel*;
do
  lsof -t $n | awk '{print($2)}' | xargs -I '{}' kill -9 {}
done
killall python3 -9
# lsof -t -i:12355 | xargs kill -9
shopt -u nullglob