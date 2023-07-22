[![PyPi](https://img.shields.io/pypi/v/videochef)](https://pypi.org/project/videochef/)

![Videochef logo](docs/logo.png)

Videochef is a Python library that makes it easier to work with videos in scientific settings. It builds on the low-level ffmpeg and pyav libraries to create a fast, Pythonic way to read and write videos.

Features include:
* VideoReader and VideoWriter classes that are fast and "just work"
* Batch processing of embarassingly parallel analyses (requires multiple CPUs)
* Easy creation of peri-event video galleries
* Precise frame counting in VideoReader makes it easy to align videos across multiple cameras that may have sparse dropped frames or slightly differing start and end points
* ..._your contributed feature here!_...

When should you use videochef vs. other python video libraries?

| You want to... you should use... | [imageio](https://github.com/imageio/imageio) | [decord](https://github.com/dmlc/decord) | videochef |
| ---          | ---     | ---    | ---       |
write video quickly... | meh | n/a (doesn't handle writing) | **yes!**
read video quickly and just extract frames... | meh | [**yes!**](https://medium.com/@haydenfaulkner/extracting-frames-fast-from-a-video-using-opencv-and-python-73b9b7dc9661) | meh
read video and run the same analysis on every frame... | meh | meh | **yes -- parallel!**
grab random / sparse frames from a video for NN training... | no | **yes!** | no
grab precise, aligned chunks of a video for peri-event analysis... | no | meh | **yes!**

## Install
`pip install videochef`, or clone the repo and `pip install -e .` from inside the repo. Don't forget to use a conda env or a venv!

## Examples

Easily read only frames that match across streams:

```python
matched_frames = [[0,2,4], [1,3,5]]  # say the second camera started 1 frame early, and each camera dropped a frame.
with VideoReader('/path/to/vid0.avi', frame_ixs=matched_frames[0],) as bottom_vid, \
    VideoReader('/path/to/vid1.avi', frame_ixs=matched_frames[1],) as top_vid:
    for i, (top_frame, bottom_frame) in enumerate(zip(top_vid, bottom_vid)):
        # do some analysis on the matched frames
```

Make peri-stimulus video galleries:
```python

stim_frames = [6000, 8020, 12100, 15001]  # some recurring stimulus or event
fps = 30
window = (-1,1)  # seconds
peri_evt_frames_list = [np.arange(fr + window[0]*fps, fr + window[1]*fps) for fr in stim_frames]

videochef.viz.peri_event_vid(
    '/path/to/vid.avi',
    '/path/to/peristim_vid.avi',
    peri_evt_frames_list=peri_evt_frames_list,
    event_frame_num_in_vid=(0 - window[0])*fps,  # event onset will be marked in the corner
    out_fps=fps/2,
)

```

Do some complex analysis on every other frame of a video, in parallel (requires multiple CPUs):
```python
def my_complex_analysis(frame):
    """Takes one video frame and analyzes it, returning an annotated frame and some scalars.

    Returns:
        A tuple!
    """
    # ... processing ...
    processed_frame = frame + 1  # dumb example
    scalar_results_dict = dict(attr1='foo', attr2='bar')
    return (processed_frame, scalar_results_dict)

step = 2
videochef.chef.video_chef(
    my_complex_analysis,
    '/path/to/my/vid.avi',
    output_types=['video', 'arrays'],  # arrays will be saved as npz's
    max_workers=3,  # ncpus - 1
    every_nth_frame=step,
    proc_suffix='_complexly_analyzed',
)

```



## Authors
Caleb Weinreb wrote the core ffmpeg code, the initial reader/writer classes, and the peri-event gallery code. Jonah Pearl wrote the parallel processing module, updated the reader class to be more efficient, and updated the writer class to "just work" with color videos. 

## Roadmap

* properly benchmark!
* figure out GPU encoding for writer

Tested and works well with:
* AVI files, encoded with `ffv1`, and pixel format `gray8`
* MP4 files, encoded with `h264`, and pixel format `yuvj420`
* Likely works well with most grayscale AVIs and color/gray MP4s. Other formats tbd.

## Citations
Logo adapted from [freepik](https://www.freepik.com/free-vector/collection-hand-drawn-chef-hats_1118072.htm#query=chef%20hat&position=8&from_view=search&track=ais)
