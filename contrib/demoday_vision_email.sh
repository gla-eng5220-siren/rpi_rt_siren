#!/bin/bash

./build/release/flame_iris \
  --libcamera 0 \
  --buzzer 26 \
  --model testdata/model \
  --webui-path webui \
  --brevo-api-key "$(cat ~/.brevo)" \
  --brevo-sender-email 'test@yccb.me' \
  --brevo-to-email 'omegacoleman@gmail.com'

