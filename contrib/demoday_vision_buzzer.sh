#!/bin/bash

./build/release/flame_iris \
  --libcamera 0 \
  --buzzer 26 \
  --model testdata/model \
  --webui-path webui

