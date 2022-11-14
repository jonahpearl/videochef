import numpy as np
from videochef.io import videoReader, videoWriter


def parallel_proc_frame(vid_path, frame_batch, reporter_val, writer_path, analysis_func=None):
    with videoReader(vid_path, np.array(frame_batch), reporter_val) as vid, videoWriter(writer_path) as writer:
        for frame in vid:
            writer.append(analysis_func(frame))