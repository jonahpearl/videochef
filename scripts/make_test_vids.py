import numpy as np
import os
from os.path import join, exists
import cv2
import videochef.io as io
import click

def generate_labeled_movie(vid_name, nframes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    with io.videoWriter(vid_name) as vid:
        for i in range(nframes):

            # write the frame number
            frame = cv2.putText(np.zeros((400, 400), dtype='uint8'), f'{i}', (200,200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Use the first row of pixels to also encode the frame number
            npix_255 = i // 255
            last_pix_val = i % 255
            frame[0, :npix_255] = 255
            frame[0, npix_255] = last_pix_val

            vid.append(frame.astype('uint8'))
    return

def generate_marked_movie(vid_name, nframes, marked_frame=2221):
    font = cv2.FONT_HERSHEY_SIMPLEX
    assert marked_frame < nframes
    with io.videoWriter(vid_name) as vid:
        for i in range(nframes):

            # write the frame number
            frame = cv2.putText(np.zeros((400, 400), dtype='uint8'), f'{i}', (200,200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Use the first row of pixels to also encode the frame number
            if i == marked_frame:
                frame[0:10, :] = 255

            vid.append(frame.astype('uint8'))
    return

@click.command()
@click.argument('out_path')
@click.option('-n', '--nframes', type=int, default=10000, help='Num frames for test video')
def main(out_path, nframes):

    if not exists(out_path):
        print(f'Creating output dir {out_path}')
        os.makedirs(out_path)
    test_movie = join(out_path, './labeled_frames.avi')
    test_mp4_movie = join(out_path, './labeled_frames_mark_2221.avi')
    
    if not exists(test_movie):
        print(f'Making movie {test_movie}')
        generate_labeled_movie(test_movie, nframes)
    else:
        print('Movie already exists, skipping.')

    if not exists(test_mp4_movie):
        print(f'Making movie {test_mp4_movie}')
        generate_marked_movie(test_mp4_movie, nframes)
    else:
        print('Movie already exists, skipping.')
    
    return

if __name__== '__main__':
    main()