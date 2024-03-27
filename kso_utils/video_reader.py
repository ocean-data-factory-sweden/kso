import hashlib
import itertools
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, Hashable, Any
from typing import Tuple, Iterator

import av
import cv2
from more_itertools import first
import numpy as np
import sys

KeyType = TypeVar("KeyType")
ItemType = TypeVar("ItemType")
BGRImageArray = np.ndarray
TimeIntervalTuple = Tuple[Optional[float], Optional[float]]


def get_memory_footprint(item: Any) -> int:
    # TODO: Recurse through dataclasses
    if isinstance(item, np.ndarray):
        return item.itemsize * item.size
    else:
        return sys.getsizeof(item)  # Only returns pointer-size, no recursion


class CacheDict(Generic[KeyType, ItemType]):
    """A simple buffer that just keeps a cache of recent entries
    Example
        cache = ItemCache(buffer_len=3)
        cache[1]='aaa'
        cache[5]='bbb'
        cache[7]='ccc'
    """

    def __init__(
        self,
        buffer_length: Optional[int] = None,
        buffer_size_bytes: Optional[int] = None,
        calculate_size_once=True,
        always_allow_one_item: bool = False,
    ):
        self._buffer = OrderedDict()
        self._buffer_length = buffer_length
        self._buffer_size_bytes = buffer_size_bytes
        self._first_object_size: Optional[int] = None
        self._calculate_size_once = calculate_size_once
        self._current_buffer_size = 0
        self._always_allow_one_item = always_allow_one_item

    def _remove_oldest_item(self):
        if len(self._buffer) > 0:
            first_key = first(self._buffer.keys())
            value, itemsize = self._buffer[first_key]
            del self._buffer[first(self._buffer.keys())]
            if itemsize is not None:
                self._current_buffer_size -= itemsize

    def __setitem__(self, key: Hashable, value: ItemType) -> None:
        size = None
        if self._buffer_length is not None and len(self._buffer) == self._buffer_length:
            self._remove_oldest_item()

        if self._buffer_size_bytes is not None:
            size = (
                get_memory_footprint(value)
                if not self._calculate_size_once or self._first_object_size is None
                else self._first_object_size
            )
            while (
                len(self._buffer) > 0
                and self._current_buffer_size + size > self._buffer_size_bytes
            ):
                self._remove_oldest_item()

            self._current_buffer_size += size

            if (
                size < self._buffer_size_bytes
                or len(self._buffer) == 0
                and self._always_allow_one_item
            ):
                self._buffer[key] = value, size
        else:
            self._buffer[key] = value, size

    def __getitem__(self, key: Hashable) -> ItemType:
        if key in self._buffer:
            value, _ = self._buffer[key]
            return value
        else:
            raise KeyError(f"{key} is not in cache")

    def __contains__(self, key: Hashable):
        return key in self._buffer


def fit_image_to_max_size(image: BGRImageArray, max_size: Tuple[int, int]):
    """Make sure image fits within (width, height) max_size while preserving aspect ratio"""
    if image.shape[0] > max_size[1] or image.shape[1] > max_size[0]:
        scale_factor = min(max_size[1] / image.shape[0], max_size[0] / image.shape[1])
        return cv2.resize(src=image, dsize=None, fx=scale_factor, fy=scale_factor)
    else:
        return image


def compute_fixed_hash(array):
    return hashlib.md5(array.tostring()).hexdigest()


@dataclass
class VideoFrameInfo:
    image: BGRImageArray
    seconds_into_video: float
    frame_ix: int
    fps: float

    def get_size_xy(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def get_progress_string(self) -> str:
        return f"t={self.seconds_into_video:.2f}s, frame={self.frame_ix}"


@dataclass
class VideoReader:
    """
    The reader efficiently provides access to video frames.
    It uses pyav: https://pyav.org/docs/stable/
    Usage:
        reader = VideoReader(path=video_segment.path, use_cache=use_cache)
        # Iterate in order
        for frame in reader.iter_frames(time_interval=(1, 2)):
            cv2.imshow('frame', frame.image)
            cv2.waitKey(1)
        # Request individually
        frame = reader.request_frame(20)  # Ask for the 20th frame
        cv2.imshow('frame', frame.image)
        cv2.waitKey(1)
    Implementation:
    - Providing them in order should be FAST
    - Requesting the same frame twice or backtracking a few frames should be VERY FAST (ie - use a cache)
    - Requesting random frames should be reasonably fast (do not scan from start)
    Note: Due to bug in OpenCV with GET_PROP_POS_FRAMES
        https://github.com/opencv/opencv/issues/9053
        We use "av": conda install av -c conda-forge
    """

    def __init__(
        self,
        path: str,
        buffer_size_bytes=1024**3,
        threshold_frames_to_scan=30,
        max_size_xy: Optional[Tuple[int, int]] = None,
        use_cache: bool = True,
    ):
        self._path = os.path.expanduser(path)
        assert os.path.exists(self._path), f"Cannot find a video at {path}"

        self.container = av.container.open(self._path)
        # self.stream = self.container.streams.video[0]

        # self._cap = cv2.VideoCapture(path)
        self._frame_cache: CacheDict[int, VideoFrameInfo] = CacheDict(
            buffer_size_bytes=buffer_size_bytes, always_allow_one_item=True
        )
        self._next_index_to_be_read: int = 0
        self._threshold_frames_to_scan = threshold_frames_to_scan
        # self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(self.container.streams.video[0].guessed_rate)

        self._n_frames = self.container.streams.video[0].frames
        self._cached_last_frame: Optional[VideoFrameInfo] = (
            None  # Helps fix weird bug... see notes below
        )
        self._max_size_xy = max_size_xy
        self._use_cache = use_cache
        self._iterator = self._iter_frame_data()

    def get_n_frames(self) -> int:
        return self._n_frames

    def time_to_nearest_frame(self, t: float) -> int:
        return max(0, min(self.get_n_frames() - 1, round(t * self._fps)))

    def frame_index_to_nearest_frame(self, index: int) -> int:
        return max(0, min(self.get_n_frames() - 1, index))

    def iter_frame_ixs(
        self,
        time_interval: TimeIntervalTuple = (None, None),
        frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Iterator[int]:
        assert time_interval == (None, None) or frame_interval == (
            None,
            None,
        ), "You can provide a time interval or frame inteval, not both"
        if time_interval != (None, None):
            tstart, tstop = time_interval
            return range(
                self.time_to_nearest_frame(tstart) if tstart is not None else 0,
                (
                    self.time_to_nearest_frame(tstop) + 1
                    if tstop is not None
                    else self.get_n_frames()
                ),
            )
        elif frame_interval != (None, None):
            istart, istop = frame_interval
            return range(
                self.frame_index_to_nearest_frame(istart) if istart is not None else 0,
                (
                    self.frame_index_to_nearest_frame(istop)
                    if istop is not None
                    else self.get_n_frames()
                ),
            )
        else:
            return range(0, self.get_n_frames())

    def iter_frames(
        self,
        time_interval: TimeIntervalTuple = (None, None),
        frame_interval: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Iterator[VideoFrameInfo]:
        for i in self.iter_frame_ixs(
            time_interval=time_interval, frame_interval=frame_interval
        ):
            yield self.request_frame(i)

    def _iter_frame_data(self):
        for frame in self.container.decode(self.container.streams.video[0]):
            yield frame

    def request_frame(self, index: int) -> VideoFrameInfo:
        """
        Request a frame of the video.  If the requested frame is out of bounds, this will return the frame
        on the closest edge.
        """
        if index < 0:
            index = self.get_n_frames() + index
        index = max(0, min(self.get_n_frames() - 1, index))

        if index == self.get_n_frames() - 1 and self._cached_last_frame is not None:
            return (
                self._cached_last_frame
            )  # There's a weird bug preventing us from loading the last frame again
        elif index in self._frame_cache:
            return self._frame_cache[index]
        elif 0 <= index - self._next_index_to_be_read < self._threshold_frames_to_scan:
            frame = None
            for _ in range(self._next_index_to_be_read, index + 1):
                try:
                    frame_data = next(self._iterator)
                except StopIteration:
                    raise Exception(
                        f"Could not get frame at index {index}, despite n_frames being {self.get_n_frames()}"
                    )

                image = frame_data.to_rgb().to_ndarray(format="bgr24")

                if self._max_size_xy is not None:
                    image = fit_image_to_max_size(image, self._max_size_xy)
                frame = VideoFrameInfo(
                    image=image,
                    seconds_into_video=self._next_index_to_be_read / self._fps,
                    frame_ix=self._next_index_to_be_read,
                    fps=self._fps,
                )
                if self._use_cache:
                    self._frame_cache[frame.frame_ix] = frame
                self._next_index_to_be_read += 1

            assert frame is not None, f"Error loading video frame at index {index}"
            return frame
        else:
            max_seek_search = 100
            stream = self.container.streams.video[0]
            pts = int(index * stream.duration / stream.frames)
            self.container.seek(pts, stream=stream)
            self._iterator = self._iter_frame_data()
            for j, f in enumerate(self._iterator):
                if j > max_seek_search:
                    raise RuntimeError(
                        f"Did not find target within {max_seek_search} frames of seek"
                    )
                if f.pts >= pts - 1:
                    self._iterator = itertools.chain([f], self._iterator)
                    break
            self._next_index_to_be_read = index
            return self.request_frame(index)


def test_video_reader(video_path: str, show=False):
    for use_cache in (False, True):
        reader = VideoReader(path=video_path, use_cache=use_cache)

        all_frame_hashes = [compute_fixed_hash(f.image) for f in reader.iter_frames()]
        all_frame_hashes_again = [
            compute_fixed_hash(f.image) for f in reader.iter_frames()
        ]
        assert all_frame_hashes == all_frame_hashes_again

        # With just the top lines:
        # Starting block 'Running with use_cache=False...
        #   EasyProfile: Running with use_cache=False took 2260.51ms
        # Starting block 'Running with use_cache=True...
        # EasyProfile: Running with use_cache=True took 1942.80ms
        count = 0
        time_snoppet_hashes = []
        for frame in reader.iter_frames(time_interval=(1, 2)):
            if show:
                cv2.imshow("frame", frame.image)
                cv2.waitKey(1)
            time_snoppet_hashes.append(compute_fixed_hash(frame.image))
            count += 1
        start_ix = reader.time_to_nearest_frame(1)
        assert count > 20
        assert all_frame_hashes[start_ix : start_ix + count] == time_snoppet_hashes

        index_snippet_hashes = []

        for i, frame in enumerate(reader.iter_frames(frame_interval=(10, 20))):
            if show:
                cv2.imshow("frame", frame.image)
                cv2.waitKey(1)
            index_snippet_hashes.append(compute_fixed_hash(frame.image))
            assert frame.frame_ix == 10 + i
        assert len(index_snippet_hashes) == 10
        assert all_frame_hashes[10:20] == index_snippet_hashes

        last_frame = reader.request_frame(-1)
        assert last_frame.frame_ix == reader.get_n_frames() - 1
        assert compute_fixed_hash(last_frame.image) == all_frame_hashes[-1]
        assert (
            reader.request_frame(reader.get_n_frames() - 1).frame_ix
            == reader.get_n_frames() - 1
        )
        assert (
            reader.request_frame(reader.get_n_frames()).frame_ix
            == reader.get_n_frames() - 1
        )

        second_last_frame = reader.request_frame(reader.get_n_frames() - 2)
        assert second_last_frame.frame_ix == reader.get_n_frames() - 2
        assert compute_fixed_hash(second_last_frame.image) == all_frame_hashes[-2]

        third_last_frame = reader.request_frame(reader.get_n_frames() - 3)
        assert third_last_frame.frame_ix == reader.get_n_frames() - 3
        assert compute_fixed_hash(third_last_frame.image) == all_frame_hashes[-3]

        last_frame_again = reader.request_frame(-1)
        assert compute_fixed_hash(last_frame_again.image) == all_frame_hashes[-1]

        first_frame = reader.request_frame(0)
        assert first_frame.frame_ix == 0
        assert compute_fixed_hash(first_frame.image) == all_frame_hashes[0]

        random_frame = reader.request_frame(20)
        assert random_frame.frame_ix == 20
        assert compute_fixed_hash(random_frame.image) == all_frame_hashes[20]
