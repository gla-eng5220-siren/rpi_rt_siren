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

~~~
sudo apt-get install cmake g++ libjpeg-dev libv4l-dev libcatch2-dev libxnnpack-dev libpthreadpool-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev pkg-config libssl-dev libgpiod-dev libcamera-dev
~~~

If you don't install the suggested version of XNNPACK or Catch2, it will be automatically downloaded and compiled. Beware this may take very long.

### Building

~~~
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build
cmake --build build
~~~

### Running

~~~
./build/release/flame_iris --libcamera 0 --model testdata/model --alarm-stdout
~~~

You will see output like:

~~~
Visual LOGIT: 8.25474 THRESHOLD: 0 [NO FIRE]
Visual LOGIT: 8.3169 THRESHOLD: 0 [NO FIRE]
Visual LOGIT: 8.30164 THRESHOLD: 0 [NO FIRE]
...
~~~

A few caveats (temporarily):

* For now, the program will continuously output detection results to stdout. We will support the buzzer and sound alarm in the future.

## Affiliation

<p align="center"><img src="/imgs/uofg_logo.png" height="200"></p>

