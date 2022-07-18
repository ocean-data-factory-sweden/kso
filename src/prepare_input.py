import os
import time
from collections.abc import Callable
import cv2 as cv
import numpy as np
from pathlib import Path
from kso_utils.koster_utils import unswedify

# globals
frame_device = cv.cuda_GpuMat()

def clearImage(frame: np.ndarray):
    """
    We take the maximum value of each channel, and then take the minimum value of the three channels.
    Then we blur the image, and then we take the maximum value of the blurred image and the value 0.5.
    Then we take the maximum value of the difference between the channel and the maximum value of the
    channel, divided by the blurred image, and the maximum value of the channel. Then we divide the
    result by the maximum value of the channel and multiply by 255
    
    :param frame: the image to be processed
    :return: The clear image
    """
    channels = cv.split(frame)
    # Get the maximum value of each channel
    # and get the dark channel of each image
    # record the maximum value of each channel
    a_max_dst = [float("-inf")] * len(channels)
    for idx in range(len(channels)):
        a_max_dst[idx] = channels[idx].max()

    dark_image = cv.min(channels[0], cv.min(channels[1], channels[2]))

    # Gaussian filtering the dark channel
    dark_image = cv.GaussianBlur(dark_image, (25, 25), 0)

    image_t = (255.0 - 0.95 * dark_image) / 255.0
    image_t = cv.max(image_t, 0.5)

    # Calculate t(x) and get the clear image
    for idx in range(len(channels)):
        channels[idx] = (
            cv.max(
                cv.add(
                    cv.subtract(channels[idx].astype(np.float32), int(a_max_dst[idx]))
                    / image_t,
                    int(a_max_dst[idx]),
                ),
                0.0,
            )
            / int(a_max_dst[idx])
            * 255
        )
        channels[idx] = channels[idx].astype(np.uint8)

    return cv.merge(channels)


def ProcFrames(proc_frame_func: Callable, frames_path: str):
    """
    It takes a function that processes a single frame and a path to a folder containing frames, and
    applies the function to each frame in the folder
    
    :param proc_frame_func: The function that will be applied to each frame
    :type proc_frame_func: Callable
    :param frames_path: The path to the directory containing the frames
    :type frames_path: str
    :return: The time it took to process all the frames in the folder, and the number of frames
    processed.
    """
    start = time.time()
    files = os.listdir(frames_path)
    for f in files:
        if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            if os.path.exists(str(Path(frames_path, f))):
                new_frame = proc_frame_func(cv.imread(str(Path(frames_path, f))))
                cv.imwrite(str(Path(frames_path, f)), new_frame)
            else:
                new_frame = proc_frame_func(
                    cv.imread(unswedify(str(Path(frames_path, f))))
                )
                cv.imwrite(str(Path(frames_path, f)), new_frame)
    end = time.time()
    return (end - start) * 1000 / len(files), len(files)


def ProcVid(proc_frame_func: Callable, vidPath: str):
    """
    It takes a function that processes a frame and a video path, and returns the average time it takes
    to process a frame and the number of frames in the video
    
    :param proc_frame_func: This is the function that will be called on each frame
    :type proc_frame_func: Callable
    :param vidPath: The path to the video file
    :type vidPath: str
    :return: The average time to process a frame in milliseconds and the number of frames processed.
    """
    cap = cv.VideoCapture(vidPath)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
        return
    n_frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            n_frames += 1
            proc_frame_func(frame)
        else:
            break
    end = time.time()
    cap.release()
    return (end - start) * 1000 / n_frames, n_frames


def ProcFrameCuda(frame: np.ndarray, size=(416, 416), use_gpu=False):
    """
    It takes a frame, resizes it to a smaller size, converts it to RGB, and then clears it
    
    :param frame: the frame to be processed
    :type frame: np.ndarray
    :param size: the size of the image to be processed
    :return: the processed frame.
    """
    if use_gpu:
        frame_device.upload(frame)
        frame_device_small = cv.resize(frame_device, dsize=size)
        fg_device = cv.cvtColor(frame_device_small, cv.COLOR_BGR2RGB)
        fg_host = fg_device.download()
        fg_host = clearImage(fg_device)
        return fg_host
    else:
        frame_device_small = cv.resize(frame, dsize=size)
        fg_device = cv.cvtColor(frame_device_small, cv.COLOR_BGR2RGB)
        fg_host = clearImage(fg_device)
        return fg_host
