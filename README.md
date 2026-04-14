<p align="center"><img src="/imgs/logo.png" height="250"></p>


# FlameIris: End-to-end Vision-led Real-time Fire Detection and Alarming

**FlameIris** is an end-to-end vision-led fire detection solution, organized and optimized for real-time edge-based deployment. Our primary target platform is *Raspberry Pi 5 4GB*. [ShuffleNetV2OnFire](https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn) is deployed and optimized for size and speed. It works with sensors and alarms in a real-time architecture.

## Highlights

- **Vision-Based**: Detect fire with a webcam!
- **Configurable Threshold**: Need these alarms to be a little more or less sensitive? No problem!
- **Real Time**: Faster detection, faster alarming, low latency
- **Inference Optimized**: BatchNorm and ReLU fused into Conv layers
- **End-to-end Solution**: From V4L capture to alarming

## Documents

### Tutorials

- [1. Setting Up Raspberry Pi](wiki/1.-Setting-Up-Raspberry-Pi.md)
- [2. Quick Start](wiki/2.-Quick-Start.md)
- [3. Alarming Methods](wiki/3.-Alarming-Methods.md)
- [4. Camera Sensors](wiki/4.-Camera-Sensors.md)
- [5. Temperature Sensors](wiki/5.-Temperature-Sensors.md)
- [6. SOLID Principles](wiki/6.-SOLID-Principles.md)
- [7. Latency Assessment](wiki/7.-Latency-Assessment.md)

[Doxygen API Documents](https://gla-eng5220-siren.github.io/docs/)

## Demo Screenshots

![WebUI](/imgs/webui.png)

![Email Alarming](/imgs/email.png)

## Building

### Install dependencies

FlameIris depends on the following pre-installed libraries:

- libjpeg
- libcamera
- libv4l
- libgpiod
- catch2
- xnnpack
- FFMPEG
- OpenSSL

CMake and at least one working C++ compiler are required as well.

Instructions for Debian or Raspbian Trixie:

```
sudo apt-get install cmake g++ pkg-config \
libjpeg-dev libv4l-dev libcatch2-dev \
libxnnpack-dev libpthreadpool-dev \
libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
libssl-dev libgpiod-dev libcamera-dev
```

If you don't install the suggested version of XNNPACK or Catch2, it will be automatically downloaded and compiled. Beware this may take very long.

### Building

```
mkdir -p build/release
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build/release
cmake --build build/release
```

### Running

Some example setup:

```
./build/release/flame_iris --mock-temp --alarm-stdout

./build/release/flame_iris \
  --libcamera 0 \
  --buzzer 26 \
  --model testdata/model \
  --webui-path webui
```

See [Quick Start](/wiki/2.-Quick-Start.md) for more usage info.

## Division of Labour

- [@omegacoleman] (https://github.com/omegacoleman), system design, vision-based detection
- [@lbqwq] (https://github.com/lbqwq), NTC temperature sensor software
- [@TamDiMan] (https://github.com/TamDiMan), Buzzer alarming method
- [@keleguan553-eng] (https://github.com/keleguan553-eng), NTC temperature sensor hardware
- [@tanglijie0408-droid] (https://github.com/tanglijie0408-droid), documentation

## Credits

Following projects are used as source-redistribution

- [libi2c](https://github.com/amaork/libi2c), MIT License
- [nlohmann/json](https://github.com/nlohmann/json), MIT License
- [cpp-httplib](https://github.com/yhirose/cpp-httplib), MIT License
- [argparse](https://github.com/p-ranav/argparse), MIT License
- [base64](https://github.com/tobiaslocker/base64), MIT License

The MIT License is compatible with out license for source redistribution.

Following projects are used as dependencies:

- libjpeg, Libjpeg License
- libcamera, LGPL-2.1-or-later
- libv4l, LGPL-2.1-or-later
- libgpiod, LGPL-2.1-or-later
- catch2, BSL-1.0 License
- xnnpack, BSD License
- FFMPEG, LGPL-2.1-or-later and GPL-2
- OpenSSL, Apache License v2 (3.0 or after), or dual OpenSSL and SSLeay license (others)

Many thanks to all these open source projects!

## Affiliation

<p align="center"><img src="/imgs/uofg_logo.png" height="200"></p>

