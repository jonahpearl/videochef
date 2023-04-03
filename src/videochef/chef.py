from os.path import join, dirname, exists, basename, splitext
from os import mkdir, listdir, remove
import numpy as np
from tqdm.contrib.concurrent import process_map
from itertools import repeat
from functools import partial
from videochef.io import VideoReader, VideoWriter
from videochef.util import make_batch_sequence, count_frames, unwrap_dictionary

import pdb

def parallel_proc_frame(
    vid_path, 
    video_reader_kwargs, 
    writer_path, 
    frame_by_frame_kwarg_dicts, 
    analysis_func=None,
    overwrite_tmp=True):
    """Helper function to pipe + process the video frames from one vid to another.
        
        NB: all args must be positional except analysis func for process_map to work correctly. 

    Arguments:
        vid_path {str} -- path to the video
        video_reader_kwargs {dict} -- kwargs to pass to VideoReader
        writer_path {str} -- where to write the processed video
        frame_by_frame_kwarg_dicts -- list of dictionaries of frame-by-frame kwargs (eg [{'key1': val1, 'key2': val1, ...}, {'key1': val2, 'key2': val2, ...}, ...])
        
    Keyword Arguments:
        analysis_func {function} -- the processing function. Must accept one video frame as a sole positional arg. Must return either a frame (if output_type is 'video') or a dictionary of scalars (if output_type is 'arrays') (default: {None}

    Raises:
        ValueError: if no analysis function is provided.
    """
    if analysis_func is None:
        raise ValueError('Please provide an analysis function!')
    
    if not overwrite_tmp and exists(writer_path):
        return
        
    output_ext = splitext(writer_path)[1]
    batch_len = len(video_reader_kwargs['frame_ixs'])
    movie_types = ['.avi', '.mp4']
    if output_ext in movie_types:
        with VideoReader(vid_path, **video_reader_kwargs) as vid, VideoWriter(writer_path) as writer:
            for iFrame, (frame, frame_kwarg_dict) in enumerate(zip(vid, frame_by_frame_kwarg_dicts)):
                writer.append(analysis_func(frame, **frame_kwarg_dict))
    elif output_ext == '.npz':
        output = {}
        with VideoReader(vid_path, **video_reader_kwargs) as vid:
            for iFrame, (frame, frame_kwarg_dict) in enumerate(zip(vid, frame_by_frame_kwarg_dicts)):
                results = analysis_func(frame, **frame_kwarg_dict)
                if iFrame == 0:
                    for key in results.keys():
                        output[key] = np.zeros((batch_len, *results[key].shape))
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
    truncate_to_n_batches=None,
    vid_read_reporter=False,
    video_reader_kwargs=None, 
    tmp_dir=None, 
    overwrite_tmp=True,
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
        truncate_to_n_batches {int} -- if not None, only process this many batches (for debugging) {default: None}
        vid_read_reporter {bool} -- if True, workers will report which video frames they're reading (mostly for debugging) (default: {False})
        tmp_dir {[type]} -- where to store the temporary (unstitched) processed videos (default: {path_to_vid/tmp})
        overwrite_tmp {bool} -- if True, remove anything in the tmp dir before starting; if False, try to use anything pre-existing (default: {True})
        proc_suffix {str} -- suffix for the outputs (default: {'_PROC'})

    Returns:
        [str] -- path to the stitched and processed video
    """

    nframes = count_frames(path_to_vid)
    batch_seq = make_batch_sequence(nframes, frame_batch_size, 0, offset=0, step=every_nth_frame)
    if truncate_to_n_batches is not None:
        batch_seq = batch_seq[:truncate_to_n_batches]
    n_expected_frames = sum([len(list(r)) for r in batch_seq])
    vid_name, vid_ext = splitext(basename(path_to_vid))
    vid_dir = dirname(path_to_vid)


    if tmp_dir is None:
       tmp_dir = join(vid_dir, 'tmp')
     
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    elif overwrite_tmp:
        for file in listdir(tmp_dir):
            remove(join(tmp_dir, file))

    if video_reader_kwargs is None:
        video_reader_kwargs = {}

    print('Processing in parallel...')

    ### Prep iters for process_map ###
    # NB, all are length of batch_seq

    # Get video writers to use  #TODO: get extension from passed in vid, so can use mp4, eg
    if output_type == 'video':
        parallel_writer_names = [join(tmp_dir, f'proc_{i}.avi') for i in range(len(batch_seq))]
    elif output_type == 'arrays':
        parallel_writer_names = [join(tmp_dir, f'proc_{i}.npz') for i in range(len(batch_seq))]
    
    # Set up kwargs for video readers
    if vid_read_reporter:
        reporter_vals = (i for i in range(len(batch_seq)))
    else:
        reporter_vals = (None for i in range(len(batch_seq)))
    partial_reader_kwarg_lists = [{'frame_ixs': np.array(batch), 'reporter_val': reporter_val, **video_reader_kwargs} for batch, reporter_val in zip(batch_seq, reporter_vals)]

    # Get chunks of frame-by-frame kwargs if needed
    if func_frame_kwargs is None:
        func_frame_kwargs = {}
    kwarg_dict_list = np.array(unwrap_dictionary(func_frame_kwargs), dtype='object')
    if len(kwarg_dict_list) == 0:
        partial_fr_by_fr_kwarg_list = [[{} for _ in batch] for batch in batch_seq]    
    elif len(kwarg_dict_list) == n_expected_frames:
        partial_fr_by_fr_kwarg_list = [kwarg_dict_list[batch] for batch in batch_seq]
    else:
        raise ValueError(f'Expected empty func_frame_kwargs or lists of length n_expected_frames ({n_expected_frames}), but got lists of length {len(kwarg_dict_list)}. Did you pass every_nth_frame correctly?')

    ### Do the parallel processing ###
    
    # Apply global kwargs
    if func_global_kwargs is None:
        func_global_kwargs = {}
    analysis_func_partial = partial(func, **func_global_kwargs)

    # Get parallel proc function
    proc_func = partial(
        parallel_proc_frame, 
        analysis_func=analysis_func_partial,
        overwrite_tmp=overwrite_tmp)

    # Call to process_map is (func, iter1, iter2, ..., iterN, **kwargs) where iter1 - iterN are iterables that are passed as positional args to func.
    # Importantly, chunksize=1 means workers get only one item from each iterable at once; nb that each single "chunk" will be frame_batch_size frames!
    process_map(proc_func,
                repeat(path_to_vid),
                partial_reader_kwarg_lists,
                parallel_writer_names,
                partial_fr_by_fr_kwarg_list,
                chunksize=1,
                max_workers=max_workers,
                total=len(batch_seq))

    if output_type == 'video':
        print('Stitching parallel videos')
        stitched_vid_name = join(tmp_dir, vid_name + proc_suffix + vid_ext)
        with VideoWriter(stitched_vid_name) as stitched_vid:
            for i, vid_name in enumerate(parallel_writer_names):
                print(f'Stitching video {i}')
                with VideoReader(vid_name) as cheffed_vid:
                    for frame in cheffed_vid:
                        stitched_vid.append(frame)

        # Check n_expected_frames matches. If not, something is wrong
        if not (n_expected_frames == count_frames(stitched_vid_name)):
            raise RuntimeError('Frame numbers in processed videos do not match. Something went wrong!')

        # Remove the tmp vids
        for file in listdir(tmp_dir):
            remove(join(tmp_dir, file))

        return stitched_vid_name

    elif output_type == 'arrays':
        print('Stitching arrays')
        stitched_npz_name = join(vid_dir, vid_name + proc_suffix + '.npz')

        # Dictionary to hold stitched np arrays
        results = {}
        counters = {}
        cheffed_npzs = [np.load(arr_name) for arr_name in parallel_writer_names]
        for iNpz, npz in enumerate(cheffed_npzs):

            # each npz contains some set of output var, each of which is a key in the npz.
            # For each output var, append it to its stitched array.
            for k in npz.keys():
                if k not in results: 
                    results[k] = np.zeros((n_expected_frames, *npz[k].shape[1:]))
                    counters[k] = 0
                len_ = len(npz[k])
                results[k][(counters[k]):(counters[k]+len_),...] = npz[k]
                counters[k] += len_
        np.savez(stitched_npz_name, **results)

        # Close the npz files
        for npz in cheffed_npzs:
            npz.close()

        # Remove the tmp arrays
        for file in listdir(tmp_dir):
            remove(join(tmp_dir, file))

        return stitched_npz_name
