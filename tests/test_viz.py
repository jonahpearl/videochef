from os.path import dirname, isfile, join, realpath
from time import sleep

import numpy as np
from videochef.util import count_frames
from videochef.viz import peri_event_vid

TEST_DATA_DIR = join(dirname(realpath(__file__)), "../test_data")


def test_peri_event_vid(tmp_path):

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, "labeled_frames.avi"))
    test_movie = join(path, "labeled_frames.avi")

    # Prep the peri-event fr lists
    peri_evt_frames_list = [np.arange(i, i + 10) for i in np.arange(0, 300, 30)]
    # out_vid = join(tmp_path, "labeled_frames_PERIEVT.mp4")
    out_vid = join(TEST_DATA_DIR, "labeled_frames_PERIEVT.mp4")

    # Make the peri-event vid
    peri_event_vid(
        test_movie,
        out_vid,
        peri_evt_frames_list,
        event_frame_num_in_vid=5,
        event_end_frame_num_in_vid=7,
        out_fps=10,
        overwrite=True,
    )

    # Not sure why there's a race condition here, but there is
    sleep(2)
    nframes = count_frames(out_vid)
    assert nframes == 10

def test_peri_event_vid_scaled(tmp_path):

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, "labeled_frames.avi"))
    test_movie = join(path, "labeled_frames.avi")

    scale = "640:-2"

    # Prep the peri-event fr lists
    peri_evt_frames_list = [np.arange(i, i + 10) for i in np.arange(0, 300, 30)]
    # out_vid = join(tmp_path, "labeled_frames_PERIEVT.mp4")
    out_vid = join(TEST_DATA_DIR, "labeled_frames_PERIEVT_scaled.mp4")

    # Make the peri-event vid
    peri_event_vid(
        test_movie,
        out_vid,
        peri_evt_frames_list,
        event_frame_num_in_vid=5,
        event_end_frame_num_in_vid=7,
        scale=scale,
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
