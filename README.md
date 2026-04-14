<p align="center"><img src="/imgs/logo.png" height="250"></p>


# FlameIris: End-to-end Vision-led Real-time Fire Detection and Alarming

**WARNING**: Development is actively in process as part of a course project. Some functions are incomplete and still in development.

**FlameIris** is an end-to-end vision-led fire detection solution, organized and optimized for real-time edge-based deployment. Our primary target platform is *Raspberry Pi 5 4GB*. [ShuffleNetV2OnFire](https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn) is deployed and optimized for size and speed. It works with sensors and alarms in a real-time architecture.

## Highlights

- **Vision-Based**: Detect fire with a webcam!
- **Configurable Threshold**: Need these alarms to be a little more or less sensitive? No problem!
- **Real Time**: Faster detection, faster alarming, low latency
- **Inference Optimized**: BatchNorm and ReLU fused into Conv layers
- **End-to-end Solution**: From V4L capture to alarming

## Documents

### Tutorials

[1. Setting Up Raspberry Pi](wiki/1.-Setting-Up-Raspberry-Pi.md)
[2. Quick Start](wiki/2.-Quick-Start.md)
[3. Alarming Methods](wiki/3.-Alarming-Methods.md)
[4. Camera Sensors](wiki/4.-Camera-Sensors.md)
[5. Temperature Sensors](wiki/5.-Temperature-Sensors.md)
[6. SOLID Principles](wiki/6.-SOLID-Principles.md)
[7. Latency Assessment](wiki/7.-Latency-Assessment.md)

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

## Affiliation

<p align="center"><img src="/imgs/uofg_logo.png" height="200"></p>

