from os.path import join, dirname, exists, basename, splitext
from os import mkdir
import numpy as np
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from functools import partial
from videochef.io import videoReader, videoWriter
from videochef.util import make_batch_sequence, count_frames


def parallel_proc_frame(vid_path, frame_batch, reporter_val, writer_path, analysis_func=None):
    with videoReader(vid_path, np.array(frame_batch), reporter_val) as vid, videoWriter(writer_path) as writer:
        for frame in vid:
            writer.append(analysis_func(frame))
            

def video_chef(func, path_to_vid, max_workers=3, frame_batch_size=500, vid_read_reporter=False, tmp_dir=None, proc_suffix='_PROC'):
    nframes = count_frames(path_to_vid)
    batch_seq = make_batch_sequence(nframes, frame_batch_size, 0)
    vid_name, vid_ext = splitext(basename(path_to_vid))
    vid_dir = dirname(path_to_vid)

    if tmp_dir is None:
        tmp_dir = join(vid_dir, 'tmp')
        mkdir(tmp_dir)
    else:
        assert exists(tmp_dir)

    print('Processing in parallel...')
    parallel_writer_names = [join(tmp_dir, f'proc_{i}.avi') for i in range(len(batch_seq))]
    if vid_read_reporter:
        reporter_vals = (i for i in range(len(batch_seq)))
    else:
        reporter_vals = (None for i in range(len(batch_seq)))
    process_map(partial(parallel_proc_frame, analysis_func=func),
                repeat(path_to_vid),
                batch_seq,
                reporter_vals,
                parallel_writer_names,
                chunksize=1,
                max_workers=max_workers,
                total=len(batch_seq))

    print('Stitching parallel videos')
    stitched_vid_name = join(tmp_dir, vid_name + proc_suffix + vid_ext)
    with videoWriter(stitched_vid_name) as stitched_vid:
        for i, vid_name in enumerate(parallel_writer_names):
            print(f'Stitching video {i}')
            with videoReader(vid_name) as cheffed_vid:
                for frame in cheffed_vid:
                    stitched_vid.append(frame)

    return stitched_vid_name
