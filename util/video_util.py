import cv2
import numpy as np
import psutil
import config
from .image_util import encode_img

def load_vid(vid_path, start_frame, end_frame):

    '''
    load video into memory

    Parameters
    ----------
    vid_path : string
    	the video name
    start_frame : int
    	the starting frame
    end_frame : int
    	the ending frame
    '''

    # create data
    print("Loading Video")

    # frames = np.zeros((end_frame-start_frame+1, 1080, 1920, 3), dtype=np.uint8)
    frames = []

    # Read in the Particular Video
    vidcap = cv2.VideoCapture(vid_path)
    fps = int(np.round(vidcap.get(cv2.CAP_PROP_FPS)))

    assert(fps in [25, 30, 50, 60])
    if fps in [50, 60]: # if fps is either 50 or 60, read every 2 frames, this applies to HVT and STT races
        frame_step = 2
    else:
        frame_step = 1 # if the fps is either 25 or 30, read every 1 frame, this applies to Kranji races

    print(f"fps: {fps}, frame step:{frame_step}")
    success,image = vidcap.read()
    count = 0

    while success:
        if count > end_frame * frame_step:
            break

        if count < start_frame * frame_step:
            success, image = vidcap.read()
            count += 1
            continue

        # frames[int((count - start_frame * frame_step)/frame_step), :, :, :] = cv2.resize(image, (1920, 1080)) # resize the image to (1920, 1080)
        img = cv2.resize(image, (1920, 1080))
        img_str = encode_img(img)
        frames.append(img_str)
   
        # check the RAM consumption, raise memory error if the remaining memory is lower than 10%
        if count%500 == 0 and config.debug:
            mem_remain_percentage = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
            print(f"Memory remaining: {mem_remain_percentage}")
            if mem_remain_percentage < 30:
                raise MemoryError("Out of RAM")

        for _ in range(frame_step):
            success,image = vidcap.read()
        count += frame_step

    print("Video Loaded")

    return frames


def find_fps(vid_path):

    '''
    get the actual fps. The actual fps == the default fps/frame_step. Please read the doc in load_vid() to understand the frame_step parameter

    Parameters
    ----------
    vid_path : string
    	path to the video file

    Returns
    -------
    fps : int
    	the actual fps of the frame set
    '''

    vidcap = cv2.VideoCapture(vid_path)
    fps = int(np.round(vidcap.get(cv2.CAP_PROP_FPS)))
    assert(fps in [25, 30, 50, 60])
    if fps in [50, 60]:
        return int(np.round(fps/2))
    return int(np.round(fps))
