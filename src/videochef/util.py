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

    if overlap >= chunk_size:
        raise ValueError('Overlap must be less than chunk size')
    
    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i:i + chunk_size])
    return out


def unwrap_dictionary(dict_):
    """Take a dictionary of form {'a': vals, 'b': vals, ...} and convert to a list of form [{'a': val1, 'b': val2, ...}, {'a': val2, 'b': val2, ...}, ...]
        Vals must all be same length!
        
    Arguments:
        dict_ {[type]} -- [description]
    """

    # Transform kwarg dict from {'a': vals, 'b': vals, ...} into [{'a': val1, 'b': val1,...}, {'a': val2, 'b': val2,...}, ...]
    tmp_lists = []  # Intermdiate step: first unpack each value for each dict [[{'a':val1, 'a': val2}], [{'b': val1}, {'b':val2}], ...]
    for key in dict_.keys():
        kwarg_lists.append([{key:v} for v in dict_[key]])
    
    # Then put into desired final form
    final_list = []
    for dicts in zip(*tmp_lists):
        final_list.append({k:v for d in dicts for k,v in d.items()})

    return final_list

def dummy_process(frame):
    # Normalize the frame's pixel vals around half max brightness (just something stupid that isn't instant)
    avg = np.mean(frame)
    std = np.std(frame)
    frame = ((frame - avg)/std + 255/2).astype('uint8')
    return frame


def count_frames(file_name):
    with av.open(file_name, 'r') as reader:
        return reader.streams.video[0].frames          
