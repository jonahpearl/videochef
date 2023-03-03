from os.path import join, dirname, realpath, isfile, exists
from os import listdir, makedirs, rmdir, remove
import os
from videochef.io import videoReader, videoWriter
from videochef.util import dummy_process, dummy_process_arrays, count_frames
from videochef.chef import video_chef
import numpy as np
import pytest

import pdb

FIXTURE_DIR = join(dirname(realpath(__file__)), '../test_data')


#TODO: make test for global kwargs + frame by frame kwargs

@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames.avi'))
def test_compare_serial_and_cheffed_labeled_avi(datafiles):

    try:
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

    finally:
        # Clean up
        for file in listdir(tmp_dir):
            if exists(join(tmp_dir, file)):
                remove(join(tmp_dir, file))
        rmdir(tmp_dir)

@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames.avi'))
def test_array_func(datafiles):
    
    try:
        # Set up
        path = str(datafiles)
        assert len(listdir(path)) == 1
        assert isfile(join(path, 'labeled_frames.avi'))
        test_movie = join(path, 'labeled_frames.avi')
        tmp_dir = join(dirname(realpath(__file__)), 'tmp2')  # TODO: update this with some pytest magic so can have multipel tests not interfere with each other
        makedirs(tmp_dir)

        # Do the processing with a func that returns a dict of values
        # will return: {'avg': avg, 'min': _min, 'max': _max, 'four_middle_px': four_middle_px}
        stitched_npz_name = video_chef(dummy_process_arrays, test_movie, tmp_dir=tmp_dir, output_type='arrays')

        npz = np.load(stitched_npz_name)
        nframes = count_frames(test_movie)
        assert nframes == len(npz['avg'])
        assert npz['four_middle_px'].shape == (nframes, 2, 2)
    
    finally:
        # Clean up
        for file in listdir(tmp_dir):
            if exists(join(tmp_dir, file)):
                remove(join(tmp_dir, file))
        os.system(f'rm -rf tmp')  # for some reason, os.rmdir fails here
        pass
