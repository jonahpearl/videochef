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
            frame = cv2.putText(np.zeros((400, 400)), f'{i}', (200,200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            vid.append(frame.astype('uint8'))
    return

@click.command()
@click.argument('output_dir', type='str', help='Directory in which to save test videos')
@click.option('-n', '--nframes', type='int', default=10000, help='Num frames for test video')
def main(out_path, nframes):

    if not exists(out_path):
        print(f'Creating output dir {out_path}')
        os.makedirs(out_path)
    test_movie = join(out_path, './labeled_frames.avi')
    
    if not exists(test_movie):
        print(f'Making movie {test_movie}')
        generate_labeled_movie(test_movie, nframes)
    else:
        print('Movie already exists, skipping.')
    
    return

if __name__== '__main__':
    main()