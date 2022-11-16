import numpy as np
import av

def make_batch_sequence(nframes, chunk_size, overlap, offset=0):
    '''
    Generates batches used to chunk videos prior to extraction.
    Parameters
    ----------
    nframes (int): total number of frames
    chunk_size (int): desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset
    Returns
    -------
    Returns list of batches
    '''

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out


def dummy_process(frame):
    # Normalize the frame's pixel vals around half max brightness (just something stupid that isn't instant)
    avg = np.mean(frame)
    std = np.std(frame)
    frame = ((frame - avg)/std + 255/2).astype('uint8')
    return frame


def count_frames(file_name):
    with av.open(file_name, 'r') as reader:
        return reader.streams.video[0].frames          
