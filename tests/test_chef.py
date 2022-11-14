from os.path import join, exists
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from videochef.io import videoReader, videoWriter
from videochef.util import dummy_process, gen_batch_sequence, count_frames
from videochef.chef import parallel_proc_frame
import numpy as np


def compare_serial_and_cheffed_labeled_avi():

    test_movie = '../data/labeled_frames.avi'
    nframes = count_frames(test_movie)
    tmp_dir = '../data/out'

    #TODO: add tmp testing dir

    # First, process it serially
    print('Processing serially...')
    serial_vid_name = join(tmp_dir, 'proc.avi')
    with videoReader(test_movie) as raw_vid, \
        videoWriter(serial_vid_name) as serial_vid:
        for frame in raw_vid:
            serial_vid.append(dummy_process(frame))

    # Then in parallel
    print('Processing in parallel...')
    max_workers = 3
    frame_chunksize = 500
    batch_seq = gen_batch_sequence(nframes, frame_chunksize, 0)
    parallel_writer_names = [join(tmp_dir, f'proc_{i}.avi') for i in range(len(batch_seq))]
    reporter_vals = (None for i in range(len(batch_seq)))
    process_map(parallel_proc_frame, repeat(test_movie), batch_seq, reporter_vals, parallel_writer_names, chunksize=1, max_workers=max_workers)
    print('Stitching parallel videos')
    stitched_vid_name = join(tmp_dir, 'stitched_proc.avi')
    with videoWriter(stitched_vid_name) as stitched_vid:
        for i, vid_name in enumerate(parallel_writer_names):
            print(f'Stitching video {i}')
            with videoReader(vid_name) as cheffed_vid:
                for frame in cheffed_vid:
                    stitched_vid.append(frame)


    assert count_frames(serial_vid_name) == count_frames(stitched_vid_name)

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