from os.path import join, dirname, realpath, isfile, exists
from os import listdir, makedirs, rmdir, remove
import os
from videochef.io import VideoReader, VideoWriter
from videochef.util import count_frames
from videochef.viz import peri_event_vid
import numpy as np
import pytest
from time import sleep

import pdb

TEST_DATA_DIR = join(dirname(realpath(__file__)), '../test_data')

def test_peri_event_vid(tmp_path):

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    # Prep the peri-event fr lists
    peri_evt_frames_list = [np.arange(i,i+10) for i in np.arange(0,300,30)]
    out_vid = join(tmp_path, 'labeled_frames_PERIEVT.avi')

    # Make the peri-event vid
    peri_event_vid(
        test_movie,
        out_vid,
        peri_evt_frames_list,
        out_fps=10,
        overwrite=True,
    )

    # Not sure why there's a race condition here, but there is
    sleep(2)
    nframes = count_frames(out_vid)
    assert nframes == 10



# if __name__ == '__main__':
#     test_movie = join('../test_data', 'labeled_frames.avi')
#     peri_evt_frames_list = [np.arange(i,i+10) for i in np.arange(0,300,30)]
#     out_vid = './tmp_perievt/labeled_frames_PERIEVT.avi'
#     peri_event_vid(
#         test_movie,
#         out_vid,
#         peri_evt_frames_list,
#         out_fps=10,
#         overwrite=True,
#     )
#     print('done')
