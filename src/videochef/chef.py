from os.path import join, dirname, exists, basename, splitext
from os import mkdir
import numpy as np
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from functools import partial
from videochef.io import videoReader, videoWriter
from videochef.util import make_batch_sequence, count_frames, unwrap_dictionary

import pdb

def parallel_proc_frame(vid_path, frame_batch, reporter_val, writer_path, kwarg_dict_list, analysis_func=None):
    """Helper function to pipe + process the video frames from one vid to another.
        
        NB: all args must be positional except analysis func for process_map to work correctly. 

    Arguments:
        vid_path {str} -- path to the video
        frame_batch {range} -- which frames to process
        reporter_val {int} -- For debugging. If True, videoReader will report which video frames its reading.
        writer_path {str} -- where to write the processed video

    Keyword Arguments:
        analysis_func {function} -- the processing function. Must accept one video frame as a sole positional arg. Must return either a frame (if output_type is 'video') or a dictionary of scalars (if output_type is 'arrays') (default: {None}
        kwarg_dict -- dictionary of frame-by-frame kwargs (eg {'key1': vals, 'key2': vals}, where vals is an array of same length as frame_batch)

    Raises:
        ValueError: if no analysis function is provided.
    """
    if analysis_func is None:
        raise ValueError('Please provide an analysis function!')
    
    output_ext = splitext(writer_path)[1]
    
    movie_types = ['.avi', '.mp4']
    if output_ext in movie_types:
        with videoReader(vid_path, np.array(frame_batch), reporter_val) as vid, videoWriter(writer_path) as writer:
            for iFrame, (frame, frame_kwarg_dict) in enumerate(zip(vid, kwarg_dict_list)):
                writer.append(analysis_func(frame, **frame_kwarg_dict))
    elif output_ext == '.npz':
        output = {}
        with videoReader(vid_path, np.array(frame_batch), reporter_val) as vid:
            for iFrame, (frame, frame_kwarg_dict) in enumerate(zip(vid, kwarg_dict_list)):
                results = analysis_func(frame, **frame_kwarg_dict)
                if iFrame == 0:
                    for key in results.keys():
                        output[key] = np.zeros((len(frame_batch), *results[key].shape))
                for key, val in results.items():
                    output[key][iFrame,...] = val
        np.savez(writer_path, **output)
    return

def video_chef(
    func, 
    path_to_vid, 
    func_global_kwargs=None, 
    func_frame_kwargs=None,
    output_type='video',
    max_workers=3, 
    frame_batch_size=500, 
    every_nth_frame=1,
    vid_read_reporter=False, 
    tmp_dir=None, 
    proc_suffix='_PROC'
    ):
    """Process a video in embarassingly parallel batches, writing out a processed video (an avi) or arrays of scalars (an npz). 
    Thin wrapper around tqdm.contrib.concurrent.process_map.

    Arguments:
        func {python function} -- the processing function. Must accept one video frame as a sole positional arg. Must return either a frame (if output_type is 'video') or a dictionary of scalars (if output_type is 'arr') (default: {None}
        path_to_vid {str} -- path to the video to be processed.

    Keyword Arguments:
        func_global_kwargs {dict} -- kwargs to pass into the analysis function once, at initialization
        func_frame_kwargs {dict} -- kwargs to pass for each frame (eg {'key1': vals, 'key2': vals}, where vals is an array of same length as the video)
        output_type {str} -- 'video' or 'arrays' (default: {'video'})
        max_workers {int} -- max_workers for process_map (default: {3})
        frame_batch_size {int} -- n frames processed per worker-batch (default: {500})
        every_nth_frame {int} -- process every nth frame (default: {1})
        vid_read_reporter {bool} -- if True, workers will report which video frames they're reading (mostly for debugging) (default: {False})
        tmp_dir {[type]} -- where to store the temporary (unstitched) processed videos (default: {path_to_vid/tmp})
        proc_suffix {str} -- suffix for the temporary (unstitched) processed videos (default: {'_PROC'})

    Returns:
        [str] -- path to the stitched and processed video
    """

    nframes = count_frames(path_to_vid)
    batch_seq = make_batch_sequence(nframes, frame_batch_size, 0, offset=0, step=every_nth_frame)
    n_expected_frames = sum([len(list(r)) for r in batch_seq])
    vid_name, vid_ext = splitext(basename(path_to_vid))
    vid_dir = dirname(path_to_vid)

    if tmp_dir is None:
        tmp_dir = join(vid_dir, 'tmp')
        mkdir(tmp_dir)
    else:
        assert exists(tmp_dir)

    print('Processing in parallel...')

    ### Prep iters for process_map ###
    # NB, all are length of batch_seq

    # Get video writers to use  #TODO: get extension from passed in vid, so can use mp4, eg
    if output_type == 'video':
        parallel_writer_names = [join(tmp_dir, f'proc_{i}.avi') for i in range(len(batch_seq))]
    elif output_type == 'arrays':
        parallel_writer_names = [join(tmp_dir, f'proc_{i}.npz') for i in range(len(batch_seq))]
    
    # Set up debugging if desired
    if vid_read_reporter:
        reporter_vals = (i for i in range(len(batch_seq)))
    else:
        reporter_vals = (None for i in range(len(batch_seq)))

    # Get chunks of frame-by-frame kwargs if needed
    if func_frame_kwargs is None:
        func_frame_kwargs = {}
    kwarg_dict_list = np.array(unwrap_dictionary(func_frame_kwargs), dtype='object')

    if len(kwarg_dict_list) == 0:
        partial_kwarg_lists = [[{} for _ in batch] for batch in batch_seq]    
    elif len(kwarg_dict_list) == nframes:
        partial_kwarg_lists = [kwarg_dict_list[batch] for batch in batch_seq]
    else:
        raise ValueError(f'Expected empty func_frame_kwargs or lists of length nframes ({nframes}), but got lists of length {len(kwarg_dict_list)}')

    ### Do the parallel processing ###
    
    # Apply global kwargs
    if func_global_kwargs is None:
        func_global_kwargs = {}
    analysis_func_partial = partial(func, **func_global_kwargs)

    # Get parallel proc function
    proc_func = partial(parallel_proc_frame, analysis_func=analysis_func_partial)

    # Call to process_map is (func, iter1, iter2, ..., iterN, **kwargs) where iter1 - iterN are iterables that are passed as positional args to func.
    # Importantly, chunksize=1 means workers get only one item from each iterable at once; nb that each single "chunk" will be frame_batch_size frames!
    process_map(proc_func,
                repeat(path_to_vid),
                batch_seq,
                reporter_vals,
                parallel_writer_names,
                partial_kwarg_lists,
                chunksize=1,
                max_workers=max_workers,
                total=len(batch_seq))

    if output_type == 'video':
        print('Stitching parallel videos')
        stitched_vid_name = join(tmp_dir, vid_name + proc_suffix + vid_ext)
        with videoWriter(stitched_vid_name) as stitched_vid:
            for i, vid_name in enumerate(parallel_writer_names):
                print(f'Stitching video {i}')
                with videoReader(vid_name) as cheffed_vid:
                    for frame in cheffed_vid:
                        stitched_vid.append(frame)

        # Check n_expected_frames matches. If not, something is wrong
        if not (n_expected_frames == count_frames(stitched_vid_name)):
            raise RuntimeError('Frame numbers in processed videos do not match. Something went wrong!')
        return stitched_vid_name

    elif output_type == 'arrays':
        print('Stitching arrays')
        stitched_npz_name = join(tmp_dir, vid_name + proc_suffix + '.npz')
        results = {}
        counters = {}
        cheffed_npzs = [np.load(arr_name) for arr_name in parallel_writer_names]
        for iNpz, npz in enumerate(cheffed_npzs):
            for k in npz.keys():
                if k not in results:  # alternatively, could pre-alloc dict on first pass?
                    results[k] = np.zeros((n_expected_frames, *npz[k].shape[1:]))
                    counters[k] = 0
                len_ = len(npz[k])
                results[k][(counters[k]):(counters[k]+len_),...] = npz[k]
                counters[k] += len_
        np.savez(stitched_npz_name, **results)
        return stitched_npz_name
