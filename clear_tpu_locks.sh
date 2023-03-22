#!/bin/bash
#god i hope this works
shopt -s nullglob
for n in /dev/accel*;
do
  lsof -t $n | awk '{print($2)}' | xargs -I '{}' kill -9 {}
done
shopt -u nullglob