import numpy as np
import av
from os.path import join, exists

import time
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from imageio import get_reader
from itertools import repeat, count
import src.videochef.io as io

import imageio as iio
import pdb


def parallel_proc_frame(vid_path, frame_batch, reporter_val, writer_path, analysis_func=process_frame):
    with io.videoReader(vid_path, np.array(frame_batch), reporter_val) as vid, io.videoWriter(writer_path) as writer:
        for frame in vid:
            writer.append(analysis_func(frame))