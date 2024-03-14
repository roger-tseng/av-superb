import io
import logging
import math
import pathlib
from typing import BinaryIO, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV, _pyav_decode_stream
from pytorchvideo.data.utils import pts_to_secs, secs_to_pts, thwc_to_cthw

logger = logging.getLogger(__name__)


class PyAVDecoder(EncodedVideoPyAV):
    def __init__(
        self,
        file_path: str,
        decode_audio: bool = True,
    ) -> None:

        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        super().__init__(
            file=video_file,
            video_name=pathlib.Path(file_path).name,
            decode_audio=decode_audio,
        )
        self._video_fps = self._container.streams.video[0].average_rate
        self._video_start = self._video_start_pts * self._video_time_base
        self._video_duration = (
            self._container.streams.video[0].duration * self._video_time_base
        )
        if self._decode_audio and self._has_audio:
            self._audio_fps = self._container.streams.audio[0].rate
            self._audio_start = self._audio_start_pts * self._audio_time_base
            self._audio_duration = (
                self._container.streams.audio[0].duration * self._audio_time_base
            )

    def get_video_clip(self, start_sec: float, end_sec: float):
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            A tensor of the clip's RGB frames with shape:
            (channel, time, height, width). The frames are of type torch.float32 and
            in the range [0 - 255].

            Returns None if no video or audio found within time range.

        """
        self._video = self._decode_video_clip(start_sec, end_sec)

        video_frames = None
        if self._video is not None:
            video_start_pts = secs_to_pts(
                start_sec, self._video_time_base, self._video_start_pts
            )
            video_end_pts = secs_to_pts(
                end_sec, self._video_time_base, self._video_start_pts
            )

            video_frames = [
                f
                for f, pts in self._video
                if pts >= video_start_pts and pts <= video_end_pts
            ]

        if video_frames is None or len(video_frames) == 0:
            logger.debug(
                f"No video found within {start_sec} and {end_sec} seconds. "
                f"Video starts at time 0 and ends at {self.duration}."
            )
            video_frames = None

        return video_frames

    def get_audio_clip(self, start_sec: float, end_sec: float) -> torch.Tensor:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            A tensor of the clip's RGB frames with shape:
            (channel, time, height, width). The frames are of type torch.float32 and
            in the range [0 - 255].

            Returns None if no video or audio found within time range.

        """
        self._audio = self._decode_audio_clip(start_sec, end_sec)

        audio_samples = None
        if self._has_audio and self._audio is not None:
            audio_start_pts = secs_to_pts(
                start_sec, self._audio_time_base, self._audio_start_pts
            )
            audio_end_pts = secs_to_pts(
                end_sec, self._audio_time_base, self._audio_start_pts
            )
            audio_samples = [
                f
                for f, pts in self._audio
                if pts >= audio_start_pts and pts <= audio_end_pts
            ]

            audio_samples = np.concatenate(audio_samples, axis=1).astype(np.float32)
            shift_pts = self._audio[0][1] - audio_start_pts
            duration_pts = audio_end_pts - audio_start_pts
            audio_samples = audio_samples[:, shift_pts : shift_pts + duration_pts]

        if audio_samples is None or len(audio_samples) == 0:
            logger.debug(
                f"No audio found within {start_sec} and {end_sec} seconds. "
                f"Video starts at time 0 and ends at {self.duration}."
            )

        return audio_samples

    def _decode_video_clip(self, start_secs: float = 0.0, end_secs: float = math.inf):
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        video_and_pts = None
        try:
            pyav_video_frames, _ = _pyav_decode_stream(
                self._container,
                secs_to_pts(start_secs, self._video_time_base, self._video_start_pts),
                secs_to_pts(end_secs, self._video_time_base, self._video_start_pts),
                self._container.streams.video[0],
                {"video": 0},
            )
            if len(pyav_video_frames) > 0:
                video_and_pts = [
                    (frame.to_image().convert("RGB"), frame.pts)
                    for frame in pyav_video_frames
                ]

        except Exception as e:
            logger.debug(f"Failed to decode video: {self._video_name}. {e}")

        return video_and_pts

    def _decode_audio_clip(self, start_secs: float = 0.0, end_secs: float = math.inf):
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        audio_and_pts = None
        try:
            if self._has_audio:
                pyav_audio_frames, _ = _pyav_decode_stream(
                    self._container,
                    secs_to_pts(
                        start_secs, self._audio_time_base, self._audio_start_pts
                    ),
                    secs_to_pts(end_secs, self._audio_time_base, self._audio_start_pts),
                    self._container.streams.audio[0],
                    {"audio": 0},
                )

                if len(pyav_audio_frames) > 0:
                    audio_and_pts = [
                        (frame.to_ndarray(), frame.pts) for frame in pyav_audio_frames
                    ]

        except Exception as e:
            logger.debug(f"Failed to decode audio: {self._video_name}. {e}")

        return audio_and_pts
