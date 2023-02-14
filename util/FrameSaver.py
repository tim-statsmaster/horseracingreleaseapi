import os
import cv2

from util.image_util import decode_img

def save_frame(horseracingResult):
    result_dir = horseracingResult.result_dir
    vid_name = horseracingResult.processing_vid_name
    frame_out_dir = os.path.join(result_dir, 'Saved_Frames', vid_name)

    if not os.path.exists(frame_out_dir):
      os.makedirs(frame_out_dir)

    start_frame, end_frame = horseracingResult.start_end_frames[vid_name]
    horseracingResult.load_frames(vid_name, start_frame, end_frame)
    
    for i, frame in enumerate(horseracingResult.frames):
        frame_idx = i + start_frame
        frame = decode_img(frame)
        cv2.imwrite(os.path.join(frame_out_dir, 'img%06d.jpg') % frame_idx, frame)
