from os.path import join, dirname, realpath, isfile
from os import listdir, makedirs, rmdir, remove
from videochef.io import videoReader, videoWriter
from videochef.util import dummy_process, count_frames
from videochef.chef import video_chef
import numpy as np
import pytest

import pdb

FIXTURE_DIR = join(dirname(realpath(__file__)), '../test_data')


@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames.avi'))
def test_precise_seek_avi(datafiles):

    # Set up
    path = str(datafiles)
    assert len(listdir(path)) == 1
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    frame_to_check = 300
    with videoReader(test_movie, np.array([frame_to_check])) as vid:
        for frame in vid:
            assert np.sum(frame[0, :]) == frame_to_check


@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames_mark_2221.mp4'))
def test_precise_seek_mp4_h264(datafiles):

    # Set up
    path = str(datafiles)
    print(datafiles)
    assert len(listdir(path)) == 1
    assert isfile(join(path, 'labeled_frames_mark_2221.mp4'))
    test_movie = join(path, 'labeled_frames_mark_2221.mp4')

    frame_to_check = 2221
    with videoReader(test_movie, np.array([frame_to_check]), mp4_to_gray=True) as vid:
        for frame in vid:
            marker_pix_val = 255  
            buffer_val = 15  # h264 compression means not all px at exactly 255, give it some buffer space.
            n_marked_rows = 10
            n_cols = 400
            assert np.sum(frame[0:10, :]) > ((marker_pix_val - buffer_val) * n_marked_rows * n_cols)  


#TODO: make test for global kwargs + frame by frame kwargs

@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames.avi'))
def test_compare_serial_and_cheffed_labeled_avi(datafiles):

    # Set up
    path = str(datafiles)
    assert len(listdir(path)) == 1
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')
    tmp_dir = join(dirname(realpath(__file__)), 'tmp')  # TODO: update this with some pytest magic so can have multipel tests not interfere with each other
    makedirs(tmp_dir)

    # First, process it serially
    print('Processing serially...')
    serial_vid_name = join(tmp_dir, 'proc.avi')
    with videoReader(test_movie) as raw_vid, \
         videoWriter(serial_vid_name) as serial_vid:
        for frame in raw_vid:
            serial_vid.append(dummy_process(frame))

    # Then in parallel
    stitched_vid_name = video_chef(dummy_process, test_movie, tmp_dir=tmp_dir)

    # Assert equal num frames
    assert count_frames(serial_vid_name) == count_frames(stitched_vid_name)

    # Check all frames are equal
    equal_frames = np.zeros(count_frames(serial_vid_name))
    non_equal_counter = 0
    with videoReader(serial_vid_name) as serial_vid, videoReader(stitched_vid_name) as stitched_vid:
        for iFrame, (serial_frame, stitched_frame) in enumerate(zip(serial_vid, stitched_vid)):
            if not np.all(serial_frame == stitched_frame):
                print(f'Frame {iFrame} not equal in serial and stitched videos')
                non_equal_counter += 1
            else:
                equal_frames[iFrame] = 1
    assert non_equal_counter == 0

    # Clean up
    for file in listdir(tmp_dir):
        remove(join(tmp_dir, file))
    rmdir(tmp_dir)
