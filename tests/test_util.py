import numpy as np
import videochef as vc
from os.path import join, dirname, realpath, isfile
from os import listdir, makedirs, rmdir, remove
import pytest

FIXTURE_DIR = join(dirname(realpath(__file__)), '../test_data')


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

@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames.avi'))
def test_count_frames_avi(datafiles):

    # Set up
    path = str(datafiles)
    assert len(listdir(path)) == 1
    assert isfile(join(path, 'labeled_frames.avi'))
    test_movie = join(path, 'labeled_frames.avi')

    nfr = vc.util.count_frames(test_movie)
    assert nfr == 7148


@pytest.mark.datafiles(join(FIXTURE_DIR, 'labeled_frames_mark_2221.mp4'))
def test_count_frames_mp4(datafiles):

    # Set up
    path = str(datafiles)
    assert len(listdir(path)) == 1
    assert isfile(join(path, 'labeled_frames_mark_2221.mp4'))
    test_movie = join(path, 'labeled_frames_mark_2221.mp4')

    nfr = vc.util.count_frames(test_movie)
    assert nfr == 10000