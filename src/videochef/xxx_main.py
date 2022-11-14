import numpy as np
import av
import os
from os.path import join, exists
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import time
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from imageio import get_reader
from itertools import repeat, count
import src.videochef.io as io

import imageio as iio
import pdb


# def parallel_dummy_process_frame(fname, frame_ixs, i):
#     print(f'starting reader {i}')
#     with io.videoReader(fname, frame_ixs) as vid:
#         for iFrame, frame in enumerate(vid):
#             if iFrame % 100 == 0:
#                 print(f'Decoded frame from reader {i}')
#             _ = dummy_process_frame(frame)


# def chunksize_test_func(chunk):
#     return (min(chunk), max(chunk))

def main():

    

    # print('Reading through video serially with no processing...')
    # takes 30 sec, (~500 it/s)
    # start_time = time.time()
    # with my_io.videoReader(test_movie, np.arange(500, 700)) as vid:
    #     for frame in tqdm(vid):
    #         pdb.set_trace()
    #         _ = dummy_process_frame(frame)
    # end_time = time.time()
    # dt = end_time - start_time
    # print(f'Dummy serial processing took {dt} seconds ({nframes/dt} it/s)')


    # time how long it takes just to read through the video in parallel
    # (This is pickling the frames before they go out to the workers)
    # takes 166 seconds, 180 it/s
    # print('Reading through video in parallel with no processing...')
    # start_time = time.time()
    # with my_io.videoReader(test_movie) as vid:
    #     process_map(dummy_process_frame, vid, chunksize=500, max_workers=6)
    # end_time = time.time()
    # dt = end_time - start_time
    # print(f'Dummy parallel processing took {dt} seconds ({nframes/dt} it/s)')

    # time how long it takes to use a generator object with vidReader and dispatch tqdm parallel it
    # takes 177 seconds, 95 of which are processing and remainder is setup/teardown.
    # runs at 190 it/s. 
    # print('Processing frames in parallel...')
    # start_time = time.time()
    # with my_io.videoReader(test_movie) as vid:
    #     process_map(process_frame, vid, chunksize=500, max_workers=6)
    # end_time = time.time()
    # print(f'Parallel processing with single reader gen took {end_time - start_time} seconds')


    # time how long it takes to get a bunch of vidReaders on the same video and have them each read a frame chunk
    # substantially slower, each one goes at about 100 it/s, but they seem to run serially!
    # all readers is started up front, but then you only get one going at a time.
    # print('Sending parallel workers to read video separately...')
    # max_workers = 6
    # frame_ixs = list(map(np.asarray, gen_batch_sequence(nframes, nframes//max_workers, 0)))
    # start_time = time.time()
    # process_map(parallel_dummy_process_frame, repeat(test_movie), frame_ixs, count(0), max_workers=max_workers)
    # end_time = time.time()
    # dt = end_time - start_time
    # print(f'Parallel processing with multiple readers took {end_time - start_time} seconds ({nframes/dt} it/s)')

    # print('Testing chunk size arg...')
    # # data = list(np.arange(1000))  # Ah, this doesn't work! The iterables are chunked for pickling, but they're still iterated individually!
    # frame_chunksize = 500
    # data = list([np.arange(i,i+frame_chunksize) for i in range(0,2000, frame_chunksize)])  # this is what we want
    # output = process_map(chunksize_test_func, data, chunksize=1, max_workers=4)
    # print(output)

    # finally, do what we're actually interested in: 
    # make a bunch of video writers, write individual videos from each process, and zip them together at the end
    # takes 60 seconds! f**k yeah. Now we hope that stitching is fast...
    # print('Reading through video in parallel and writing many outputs...')
    # start_time = time.time()
    # max_workers = 3
    # frame_chunksize = 500
    # batch_seq = gen_batch_sequence(nframes, frame_chunksize, 0)
    # parallel_writer_names = [f'./out/proc_{i}.avi' for i in range(len(batch_seq))]
    # # reporter_vals = (i for i in range(len(batch_seq)))
    # reporter_vals = (None for i in range(len(batch_seq)))
    # process_map(parallel_writing_process_frame, repeat(test_movie), batch_seq, reporter_vals, parallel_writer_names, chunksize=1, max_workers=max_workers)
    # end_time = time.time()
    # dt = end_time - start_time
    # print(f'Processing took {dt} seconds ({nframes/dt} it/s)')


    # debugging: show first frame of each parallel vid, and each supposed corresponding frame of full vid
    # these match perfectly. so the issue is in the stitching.
    # for batch, out_name in zip(batch_seq, parallel_writer_names):
    #     frame_num = np.array(int(batch.start))
    #     relative_frame_num = np.array([0])
    #     with my_io.videoReader('./out/proc.avi', frame_num) as full_vid, \
    #          my_io.videoReader(out_name, relative_frame_num) as stub_vid:        
    #          for full_frame, stub_frame in zip(full_vid, stub_vid):
    #             # plt.subplot(1,2,1)
    #             # plt.imshow(full_frame)
    #             # plt.subplot(1,2,2)
    #             # plt.imshow(stub_frame)
    #             # plt.suptitle(f'Frame {frame_num}')
    #             # plt.savefig(f'./out/debug_full_vs_stitched_frame_{frame_num}.png')

    #             if not np.all(full_frame == stub_frame):
    #                 print(f'Frame {frame_num} not equal in serial and stub videos')


    # stich the videos back together. (takes 30 seconds)
    # print('Stitching parallel videos')
    # out_vid = './out/stitched_proc.avi'
    # start_time = time.time()
    # with my_io.videoWriter(out_vid) as writer:
    #     for i, vid_name in enumerate(parallel_writer_names):
    #         print(f'Stitching video {i}')
    #         with my_io.videoReader(vid_name) as vid:
    #             for frame in vid:
    #                 writer.append(frame)
    # end_time = time.time()
    # dt = end_time - start_time
    # print(f'Processing took {dt} seconds ({nframes/dt} it/s)')



    print('Validating stitched video...')
    serial_name = './out/proc.avi'
    stitched_name = './out/stitched_proc.avi'
    
    print(f'Serial mov has {get_reader(serial_name).count_frames()} frames')
    print(f'Stitched mov has {get_reader(stitched_name).count_frames()} frames')

    equal_frames = np.zeros(get_reader(serial_name).count_frames())
    non_equal_counter = 0
    print_every_n_nonequal = 1000
    with io.videoReader(serial_name) as serial_vid, io.videoReader(stitched_name) as stitched_vid:
        for iFrame, (serial_frame, stitched_frame) in enumerate(zip(serial_vid, stitched_vid)):
            if not np.all(serial_frame == stitched_frame):
                print(f'Frame {iFrame} not equal in serial and stitched videos')
                non_equal_counter += 1

                print('Looking for matching frames...')

                # if non_equal_counter % print_every_n_nonequal == 0:
                #     plt.subplot(1,2,1)
                #     plt.imshow(serial_frame)
                #     plt.subplot(1,2,2)
                #     plt.imshow(stitched_frame)
                #     plt.suptitle('Frame {iFrame}')
                #     plt.savefig(f'./out/non_equal_frames_{iFrame}.png')
            else:
                equal_frames[iFrame] = 1
    
    plt.figure()
    plt.plot(equal_frames)
    plt.savefig('./out/equal_frames.png')

    if non_equal_counter == 0:
        print('Congratulations!! The videos match.')

    


    

if __name__ == '__main__':
    main()