import numpy as np
import av
import subprocess
import datetime
import os

import pdb

class VideoWriter():
    """A simple, ffmpeg-based video writer. Will try to infer grayscale vs. color (use RGB!), and act accordingly.
    Inputs should be (nframes) x (w) x (h) x (RGB).

    """
    def __init__(self, file_name, **write_frames_options):
        self.pipe = None
        self.file_name = file_name
        self.write_frame_options = write_frames_options
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.pipe is not None:
            self.pipe.stdin.close()
            self.pipe.wait()  # wait for ffmpeg to finish so the video is ready to go on the next line of code; otherwise run into weird conditions where video isn't saved to disk yet
        
    def append(self, frames):
        """Write frames to the file.
        Will convert frames into an array (nframes) x (w) x (h) x color, by inference.
        Frames should have color as final dimension, if at all.

        Arguments:
            frames {[type]} -- [description]
        """

        # Infer whether or not the frames have color
        # Will fail if the video is 3 px tall...seems unlikely.
        is_color = (len(frames.shape) >= 3) and (frames.shape[-1] == 3)

        # Re-shape the array accordingly
        if is_color and (len(frames.shape) == 3):
            # If color but only one frame, add a singleton dim in front
            frames = frames[None]
        elif is_color and (len(frames.shape) == 4):
            pass
        elif len(frames.shape) == 2:
            # If not color and only one frame, add a singleton dimension in front
            frames = frames[None]

        # Set default options if video is in color or gray.
        if is_color:
            pixel_format = self.write_frame_options.pop('pixel_format', 'yuv420p')  # output pixel format
            codec = self.write_frame_options.pop('codec', 'libx264')
        else:
            pixel_format = self.write_frame_options.pop('pixel_format', 'gray8')
            codec = self.write_frame_options.pop('codec', 'ffv1')

        # if type(frames) ==  np.ndarray and len(frames.shape) == 2:
        #      frames = frames[None]

        self.pipe = write_frames(
            self.file_name, 
            frames, 
            pipe=self.pipe, 
            pixel_format=pixel_format,
            codec=codec,
            **self.write_frame_options
        )

class VideoReader():
    def __init__(self, file_name, frame_ixs=None, reporter_val=None, mp4_to_gray=False):
        self.file_name = file_name
        self.file_ext = os.path.splitext(self.file_name)[1]
        self.frame_ixs = frame_ixs
        self.reporter_val = reporter_val
        self.mp4_to_gray = mp4_to_gray

        if frame_ixs is not None:
            self.first_frame_ix = int(np.min(frame_ixs))
            self.final_frame_ix = int(np.max(frame_ixs))
        else:
            self.first_frame_ix = 0
            self.final_frame_ix = None
        
        if self.reporter_val is not None:
            self.reporter_val = int(reporter_val)

    def __enter__(self):
        self.reader = av.open(self.file_name, 'r')
        self.reader.streams.video[0].thread_type = "AUTO"
        self.codec = self.reader.streams.video[0].name
        self.pix_fmt = self.reader.streams.video[0].format.name
        self.rate = self.reader.streams.video[0].average_rate
        self.time_base = self.reader.streams.video[0].time_base
        self.start_time = self.reader.streams.video[0].start_time

        # Create frame mask
        frame_mask = np.zeros(self.reader.streams.video[0].frames)
        if self.frame_ixs is not None:
            assert self.frame_ixs.max() < len(frame_mask), 'frame_ixs exceeds the video length'
            frame_mask[self.frame_ixs.astype(int)] = 1
        else: 
            frame_mask[:] = 1

        # Seek to first frame to use. When found, pass it to the frame_gen func for use as the first frame.
        if self.first_frame_ix != 0:
            self.first_frame_to_yield = self._precise_seek(self.first_frame_ix)
        else:
            self.first_frame_to_yield = None

        # Return a generator object that will decode the frames into np arrays
        return self.frame_gen(self.reader.decode(video=0), frame_mask)

    def _plain_frame_gen(self):
        for packet in self.reader.demux(self.reader.streams.video[0]):  #TODO: check if packet has time attribute? If so, can use packet time instead of frame time here so we dont hvae to decode and it'll speed it up?
            for frame in packet.decode():
                yield frame

    def _precise_seek(self, frame_num_to_seek):
        """See https://github.com/PyAV-Org/PyAV/blob/b65f5a9f93144d2eadbb3e460bb45d869f5ce2fe/scratchpad/second_seek_example.py
        """
        target_sec = frame_num_to_seek * 1 / self.rate
        target_pts = int(target_sec / self.time_base) + self.start_time
        self.reader.seek(target_pts, stream=self.reader.streams.video[0])
        for frame in self._plain_frame_gen():
            # if frame.pts >= frame_num_to_seek:  # fails on mp4s
            frame_num = frame.time * self.rate  # this seems more reliable
            if frame_num >= frame_num_to_seek:
                return frame

    def frame_gen(self, reader, frame_mask):

        # Handle initial boundary condition from precise seeking
        if self.first_frame_to_yield is not None:
            self.ix_offset = 1
            yield self.convert_frame_to_np(self.first_frame_to_yield)
        else:
            self.ix_offset = 0

        # Then yield subsequent frames
        for relative_ix, frame in enumerate(reader):
            ix = relative_ix + self.first_frame_ix + self.ix_offset # since enumerate always starts at 0, even if frame is not 0th frame
            if self.reporter_val: print(f'read frame {ix} from reader {self.reporter_val}')
            if self.final_frame_ix is not None and (ix > self.final_frame_ix): break  # short circuit to speed up multiprocessing
            if frame_mask[ix]:
                yield self.convert_frame_to_np(frame)
    
    
    def convert_frame_to_np(self, frame):
        if self.file_ext == '.avi' and self.codec == 'ffv1' and 'gray' in self.pix_fmt:
            return frame.to_ndarray()
        elif self.file_ext == '.avi' and self.codec == 'h264' and 'yuv' in self.pix_fmt:
            return np.array(frame.to_image())
        elif (self.file_ext == '.mp4' and self.codec == 'h264' and (self.pix_fmt == 'yuvj420p' or self.pix_fmt == 'yuv420p' or self.pix_fmt == 'yuv444p')) or \
             (self.file_ext == '.mp4' and (self.codec == 'h265' or self.codec == 'hevc') and (self.pix_fmt == 'yuvj420p' or self.pix_fmt == 'yuv420p')) or \
             (self.file_ext == '.mp4' and self.codec == 'mpeg4' and self.pix_fmt == 'yuv420p'):
            if self.mp4_to_gray:
                return np.array(frame.to_image())[:,:,0]
            else:
                return np.array(frame.to_image())
        else:
            raise NotImplementedError(f'Video format {self.file_ext} with codec {self.codec} and px fmt {self.pix_fmt} is not supported yet.')
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.close()            
                
def write_frames(filename, frames, 
                 threads=6, fps=30, crf=10,
                 pixel_format='gray8', codec='ffv1',
                 pipe=None, slices=24, slicecrc=1,
                 preset='veryfast'):
    """Use ffmpeg to write frames into a movie

    Arguments:
        filename {str} -- full filename of video to write
        frames {iterable of arrays} -- the frames to write. Should be numpy arrays (probably uint8's)

    Keyword Arguments:
        threads {int} -- n threads for ffmpeg (default: {6})
        fps {int} -- desired output fps (default: {30})
        crf {int} -- quality factor. Normal default is 23, here we boost it a bit (lower is better), because science. (default: {10})
        pixel_format {str} -- output pixel format. For quicktime playable movies, use "yuv420p" and pass 3D frames (). (default: {'gray8'})
        codec {str} -- video codec to use. For color, prefer "libx264" (default: {'ffv1'})
        pipe {[type]} -- internal use (default: {None})
        slices {int} -- number of ffmpeg slices (default: {24})
        slicecrc {int} -- ? (default: {1})

    Returns:
        [type] -- [description]
    """
    
    frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])

    if pixel_format == 'gray8' or 'rgb' in pixel_format:
        command = ['ffmpeg',
                '-y',
                '-loglevel', 'fatal',
                '-framerate', str(fps),
                '-f', 'rawvideo',
                '-s', frame_size,
                '-pix_fmt', pixel_format,
                '-i', '-',
                '-an',
                '-crf',str(crf),
                '-vcodec', codec,
                '-preset', preset,
                '-threads', str(threads),
                '-slices', str(slices),
                '-slicecrc', str(slicecrc),
                '-r', str(fps),
                filename]
    elif 'yuv' in pixel_format:
        # assume user is passing rgb (unlikely to be passing yuv formatted arrays)
        command = ['ffmpeg',
                '-y',
                '-loglevel', 'fatal',
                '-framerate', str(fps),
                '-f', 'rawvideo',
                '-s', frame_size,
                '-pix_fmt', 'rgb24',  # user will pass input as rgb...
                '-i', '-',
                '-an',
                '-crf',str(crf),
                '-c:v', 'libx264',
                '-pix_fmt', pixel_format,  # ...but output will be converted to yuv
                '-preset', preset,
                '-threads', str(threads),
                '-slices', str(slices),
                '-slicecrc', str(slicecrc),
                '-r', str(fps),
                filename]


    if not pipe: pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    dtype = np.uint16 if pixel_format.startswith('gray16') else np.uint8
    for i in range(frames.shape[0]): pipe.stdin.write(frames[i,...].astype(dtype).tobytes())
    return pipe


def read_frames(filename, frames, threads=6, fps=30, frames_is_timestamp=False,
                pixel_format='gray16', frame_size=(640,576),
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .mp4/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices
    Returns:
        3d numpy array:  frames x h x w
    """

    if frames_is_timestamp: start_time = str(datetime.timedelta(seconds=frames[0]))
    else: start_time = str(datetime.timedelta(seconds=frames[0]/fps))
    
    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-vsync','0',
        '-ss', start_time,
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command
    
  
    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    
    dtype = ('uint16' if '16' in pixel_format else 'uint8')
    video = np.frombuffer(out, dtype=dtype).reshape((len(frames), frame_size[1], frame_size[0]))
    return video
