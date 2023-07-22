[![PyPi](https://img.shields.io/pypi/v/videochef)](https://pypi.org/project/videochef/)

![Videochef logo](docs/logo.png)

Videochef is a Python library that makes it easier to work with videos in scientific settings. It builds on the low-level ffmpeg and pyav libraries to create a fast, Pythonic way to read and write videos.

Features include:
* VideoReader and VideoWriter classes that are fast and "just work"
* Precise frame counting in VideoReader makes it easy to align videos across multiple cameras that may have sparse dropped frames or slightly differing start and end points
* Batch processing of embarassingly parallel analyses (requires multiple CPUs)
* Easy creation of peri-event video galleries
* ..._your contributed feature here!_...

When should you use videochef vs. other python video libraries?

| You want to... you should use... | [imageio](https://github.com/imageio/imageio) | [decord](https://github.com/dmlc/decord) | videochef |
| ---          | ---     | ---    | ---       |
write video quickly... | maybe | n/a (doesn't handle writing) | yes!
read video quickly and just extract frames... | no | [**yes!**](https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661) | meh
read video and run the same analysis on every frame... | meh | meh | **yes -- parallel!**
grab random / sparse frames from my video for NN training... | no | **yes!** | no
grab precise, aligned chunks of video for peri-event analysis... | no | meh | **yes!**

## Install
`pip install videochef`, or clone the repo and `pip install -e .` from inside the repo.

## Examples
TODO

## Authors
Caleb Weinreb wrote the core ffmpeg code, the initial reader/writer classes, and the peri-event gallery code. Jonah Pearl wrote the parallel processing module, updated the reader class to be more efficient, and updated the writer class to "just work" with color videos. 

## Roadmap
Tested and works well with:
* AVI files, encoded with `ffv1`, and pixel format `gray8`
* MP4 files, encoded with `h264`, and pixel format `yuvj420`
* Likely works well with most grayscale AVIs and color/gray MP4s. Other formats tbd.

## Citations
Logo adapted from [freepik](https://www.freepik.com/free-vector/collection-hand-drawn-chef-hats_1118072.htm#query=chef%20hat&position=8&from_view=search&track=ais)
