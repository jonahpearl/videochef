import imageio as iio
from contextlib import ExitStack
from videochef.io import VideoReader, VideoWriter
from tqdm.notebook import tqdm
import cv2
import numpy as np
from os.path import join, dirname, realpath, isfile, exists
from os import listdir, makedirs, rmdir, remove

import pdb

def add_titles(
    panels, 
    titles, 
    text_origin=(5,18), 
    text_size=0.4, 
    text_color=(255,255,255)
):
    """Helper function to add titles to a set of video panels

    Arguments:
        panels {list of np.array} -- a list of 2d arrays
        titles {list of str} -- a list of the titles for each panel

    Keyword Arguments:
        text_origin {tuple} -- where to place the text on the panel (default: {(5,18)})
        text_size {float} -- cv2 text size (default: {0.4})
        text_color {tuple} -- cv2 text color (default: {(255,255,255)})

    Returns:
        panels -- panels with titles appended
    """
    
    return [
        cv2.putText(
            panel.copy(),
            title, 
            text_origin, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_size, 
            text_color, 
            1, 
            cv2.LINE_AA
        ) 
        for panel,title in zip(panels,titles)
    ]

def peri_event_vid(
    in_vid_name,
    out_vid_name,
    peri_evt_frames_list,
    event_frame_num_in_vid=None,
    traces=None,
    trace_to_vid_fs_ratio=None,
    out_fps=12,
    overwrite=False
):
    """Create a tiled gallery of peri-event videos

    Arguments:
        in_vid_name {path} -- video to read
        out_vid_name {path} -- video to write
        peri_evt_frames_list {list of iterables} -- peri-event frame lists for each event
        event_frame_num_in_vid {int} -- if passed, mark before/after each event onset with a hollow/filled circle. Otherwise not shown.

    Keyword Arguments:
        traces {array} -- TODO: allow user to display sync'd traces (default: {None})
        trace_to_vid_fs_ratio {[type]} -- TODO: ^^ (default: {None})
        out_fps {int} -- desired fps of written video. Often nice to slow it down. (default: {12})
        overwrite {bool} -- if True, overwrites any existing video at out_vid_name (default: {False})

    Returns: n/a
    """

    # Plan layout to be as close to square as possible
    num_stims = len(peri_evt_frames_list)
    nrows = np.ceil(np.sqrt(num_stims)).astype('int')

    # Skip already existing vids?
    if exists(out_vid_name) and not overwrite: 
        return None

    # One video per odor/session pair.
    waiting_writer = VideoWriter(
        out_vid_name, 
        fps=out_fps, 
        codec='h264', 
        pixel_format='rgb24'  # 8 bit, 3 colors
    )
    with ExitStack() as stack:

        # Get video readers (and they seek to exactly the right place on this line!)
        print('Getting readers...')
        peristim_readers = [
            stack.enter_context(
                VideoReader(
                    in_vid_name, 
                    np.array(frames),
                )
            ) for frames in peri_evt_frames_list
        ]

        print('getting writer')
        writer = stack.enter_context(waiting_writer)

        # Iterate through all readers in parallel ((frame0 stim0, frame0 stim1,...), (frame1 stim0, frame1 stim1,...), ...)
        # which will get condensed across stims into bigframe 0, bigframe 1, bigframe 2,...
        # ix counts aligned frame num (ie starts at 0)
        for ix, frames in tqdm(enumerate(zip(*peristim_readers)), total=len(peri_evt_frames_list[0])):
            all_rows = []

            # Add each mini-frame to the big frame
            panels,titles = [],[]  # empty lists for frame

            ii = 0  # counter for layout
            for iReader, frame in enumerate(frames):

                # Convert frame to RGB if it isn't already
                frame = np.stack(3 * (frame,), axis=-1)

                # Append a mark for whether event fr has passed or not
                if event_frame_num_in_vid is not None:
                    if (ix < event_frame_num_in_vid):
                        fill = 1
                    else:
                        fill = -1  # for some reason, -1 fills it
                    frame = cv2.circle(frame, (18,96), 15, (0,255,0), fill)

                # Mark traces if requested
                # TODO: have to mark real-time on the trace by tracking the right xval. Boundary conditions make it hard. pure midpoint only works when neither boundary is limiting.
#                 trace_start_idx = int(max(0, (ix*trace_to_vid_fs_ratio - 200)))
#                 trace_end_idx = int(min(traces.shape[1], (ix*trace_to_vid_fs_ratio + 200)))
#                 slice_to_show = slice(trace_start_idx, trace_end_idx, 2)
#                 if traces is not None:
#                     vals = traces[iReader, slice_to_show]
#                     vals_on_frame_x = np.arange(len(vals))
#                     vals_on_frame_y = 150 + -1*vals*20
#                     pts = np.column_stack([vals_on_frame_x, vals_on_frame_y]).round().astype(np.int32)
#                     frame = cv2.polylines(frame, [pts], False, (0,255,0))
#                     midpoint_idx = len(pts)//2
#                     frame = cv2.circle(frame, pts[midpoint_idx], 3, (255,0,0), fill)
    
                # Add to frame
                panels.append(frame.astype('uint8'))
                titles.append(f'Instance: {iReader}; fr: {ix}')

                # If at end of row, or end of rectangle, fill out big frame
                if ii == (nrows-1):
                    panels = add_titles(panels, titles, text_size=1, text_origin=(18,32))
#                     panels = add_titles(panels, odor_infos, text_size=1, text_origin=(18,64))
                    all_rows.append(panels)
                    ii = 0
                    panels,titles = [],[]
                elif iReader == (len(frames)-1):
                    panels = add_titles(panels, titles, text_size=1, text_origin=(18,32))
                    all_rows.append(panels)
                    ii = 0
                else:
                    ii += 1
            
            # Stack the individual panels into rows
            stacked_rows = [np.hstack(panels) for panels in all_rows]
            
            # Deal with any non-full rows
            max_row = np.argmax([row.size for row in stacked_rows])
            for iRow in range(len(stacked_rows)):
                diff = stacked_rows[max_row].shape[1] - stacked_rows[iRow].shape[1]
                if diff != 0:
                    stacked_rows[iRow] = np.pad(stacked_rows[iRow], ((0,0), (0,diff), (0,0)), constant_values=0)
            
            # Stack the rows (now all identical shapes)
            big_frame = np.vstack(stacked_rows).astype('uint8')

            # Save this frame into the video
            writer.append(big_frame)

    return