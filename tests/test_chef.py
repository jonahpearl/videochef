from os.path import join, dirname, realpath, isfile, exists
from os import listdir, makedirs, rmdir, remove
import os
from videochef.io import VideoReader, VideoWriter
from videochef.util import dummy_process, dummy_process_arrays, dummy_process_vid_and_arrays, count_frames
from videochef.chef import video_chef
import numpy as np
import pytest

import pdb


TEST_DATA_DIR = join(dirname(realpath(__file__)), '../test_data')
# FIXTURE_DIR = join(dirname(realpath(__file__)), '../test_data')


#TODO: make test for global kwargs + frame by frame kwargs


def test_compare_serial_and_cheffed_labeled_avi(tmp_path):

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')
    

    # First, process it serially
    print('Processing serially...')
    serial_vid_name = join(tmp_path, 'proc.avi')
    with VideoReader(test_movie) as raw_vid, \
        VideoWriter(serial_vid_name) as serial_vid:
        for frame in raw_vid:
            out = dummy_process(frame)  # returns a tuple of (frame,)
            serial_vid.append(out[0])

    # Then in parallel
    stitched_vid_name = video_chef(dummy_process, test_movie, tmp_dir=tmp_path, remove_chunks=False, output_types=['video'], overwrite_tmp=False)  # must pass overwrite_tmp=false here, or it will overwrite the serial version!
    print(stitched_vid_name)
    stitched_vid_name = stitched_vid_name[0] # video_chef returns a list of output paths, but we only have one output

    # Assert equal num frames
    assert count_frames(serial_vid_name) == count_frames(stitched_vid_name)

    # Check all frames are equal
    equal_frames = np.zeros(count_frames(serial_vid_name))
    non_equal_counter = 0
    with VideoReader(serial_vid_name) as serial_vid, VideoReader(stitched_vid_name) as stitched_vid:
        for iFrame, (serial_frame, stitched_frame) in enumerate(zip(serial_vid, stitched_vid)):
            if not np.all(serial_frame == stitched_frame):
                print(f'Frame {iFrame} not equal in serial and stitched videos')
                non_equal_counter += 1
            else:
                equal_frames[iFrame] = 1
    assert non_equal_counter == 0


def test_array_func(tmp_path):
    
    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    # Do the processing with a func that returns a dict of values
    # will return: {'avg': avg, 'min': _min, 'max': _max, 'four_middle_px': four_middle_px}
    stitched_npz_name = video_chef(dummy_process_arrays, test_movie, tmp_dir=tmp_path, output_types=['arrays'], remove_chunks=False)

    stitched_npz_name = stitched_npz_name[0]  # video_chef returns a list of names, but we only have one here
    npz = np.load(stitched_npz_name)
    nframes = count_frames(test_movie)
    assert nframes == len(npz['avg'])
    assert npz['four_middle_px'].shape == (nframes, 2, 2)
    

def test_vid_and_array(tmp_path):
    
    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    # Do the processing with a func that returns a dict of values
    # will return: {'avg': avg, 'min': _min, 'max': _max, 'four_middle_px': four_middle_px}
    stitched_names = video_chef(dummy_process_vid_and_arrays, test_movie, tmp_dir=tmp_path, output_types=['video', 'arrays'], remove_chunks=False, overwrite_tmp=False)

    stitched_vid_name, stitched_npz_name = tuple(stitched_names)  # video_chef returns a list of names

    npz = np.load(stitched_npz_name)
    nframes = count_frames(test_movie)
    assert nframes == len(npz['avg'])
    assert npz['four_middle_px'].shape == (nframes, 2, 2)

    assert count_frames(test_movie) == count_frames(stitched_vid_name)
