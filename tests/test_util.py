import numpy as np
import videochef as vc
from videochef.io import VideoReader, VideoWriter
from os.path import join, dirname, realpath, isfile
from os import listdir, makedirs, rmdir, remove
import pytest


TEST_DATA_DIR = join(dirname(realpath(__file__)), '../test_data')


def test_make_batch_sequence():
    nfr = 100
    chunk_size = 10
    overlap = 0
    offset = 0

    seq = vc.util.make_batch_sequence(nfr, chunk_size, overlap, offset=offset)
    assert seq[0] == range(0,10)

    seq = vc.util.make_batch_sequence(nfr, chunk_size, overlap, offset=5)
    assert seq[0] == range(5,15)
    assert seq[-1] == range(95,100)

    seq = vc.util.make_batch_sequence(nfr, chunk_size, 5, offset=5)
    assert seq[0] == range(5,15)
    assert seq[1] == range(10,20)
    assert seq[-2] == range(85,95)
    assert seq[-1] == range(90,100)


def test_count_frames_avi():

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    nfr = vc.util.count_frames(test_movie)
    assert nfr == 7148


def test_count_frames_mp4():

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames_mark_2221.mp4'))
    test_movie = join(path, 'labeled_frames_mark_2221.mp4')

    nfr = vc.util.count_frames(test_movie)
    assert nfr == 10000


def test_precise_seek_avi():

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    frame_to_check = 300
    with VideoReader(test_movie, np.array([frame_to_check])) as vid:
        for frame in vid:
            assert np.sum(frame[0, :]) == frame_to_check


def test_precise_seek_mp4_h264():

    # Set up
    path = str(TEST_DATA_DIR)
    assert isfile(join(path, 'labeled_frames_mark_2221.mp4'))
    test_movie = join(path, 'labeled_frames_mark_2221.mp4')

    frame_to_check = 2221
    with VideoReader(test_movie, np.array([frame_to_check]), mp4_to_gray=True) as vid:
        for frame in vid:
            marker_pix_val = 255  
            buffer_val = 15  # h264 compression means not all px at exactly 255, give it some buffer space.
            n_marked_rows = 10
            n_cols = 400
            assert np.sum(frame[0:10, :]) > ((marker_pix_val - buffer_val) * n_marked_rows * n_cols)  


def test_precise_seek_mp4_h265():

    # Set up
    path = str(TEST_DATA_DIR)
    test_movie = join(path, 'labeled_frames_mark_2221_H265.mp4')
    assert isfile(test_movie)
    
    frame_to_check = 2221
    with VideoReader(test_movie, np.array([frame_to_check]), mp4_to_gray=True) as vid:
        for frame in vid:
            marker_pix_val = 255  
            buffer_val = 15  # h264 compression means not all px at exactly 255, give it some buffer space.
            n_marked_rows = 10
            n_cols = 400
            assert np.sum(frame[0:10, :]) > ((marker_pix_val - buffer_val) * n_marked_rows * n_cols)  