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

## Demo Screenshots

![Detecting UofG landscape photo -- no fire detected](/imgs/no_fire.png)

![Detecting house fire picture from Dunnings 2018 Dataset -- fire detected](/imgs/fire.png)

## Building

### Install dependencies

FlameIris depends on the following pre-installed libraries:

- libjpeg
- libv4l
- catch2
- xnnpack

CMake and at least one working C++ compiler are required as well.

Instructions for Debian or Raspbian Trixie:


aa

~~~
sudo apt-get install cmake g++ libjpeg-dev libv4l-dev libcatch2-dev libxnnpack-dev pkg-config
~~~

If you don't install the suggested version of XNNPACK or Catch2, it will be automatically downloaded and compiled. Beware this may take very long.

### Building

~~~
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -S . -B build
cmake --build build
~~~

### Running

~~~
MODEL_PATH=testdata/model ./build/flame_iris
~~~

A few caveats (temporarily):

* A working webcam is expected at /dev/video0
* For now, the program will continuously output detection results to stdout. We will support the buzzer and sound alarm in the future.

## Affiliation

<p align="center"><img src="/imgs/uofg_logo.png" height="200"></p>

