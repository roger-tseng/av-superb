import math

import numpy as np
import torch
import torchaudio


def get_visual_seq(
    frames,
    start_idx,
    end_idx,
    video_num_frames,
    crop_size,
    spatial_sample_index,
):
    """
    Sample a sequence of clips from the input video, and apply
    visual transformations.
    args:
        frames (tensor): a tensor of video frames, dimension is
            `num frames` x `height` x `width` x `channel`.
        start_idx (list): list of the index of the start frame of each clip.
        end_idx (list): list of the index of the end frame of each clip.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        spatial_sample_index (int): if -1, perform random spatial sampling.
            If 0, 1, or 2, perform left, center, right crop if width is
            larger than height, and perform top, center, buttom crop if
            height is larger than width.
    returns:
        clip_seq (tensor): a sequence of sampled frames. The dimension is
            `sequence length` x `channel` x `num frames` x `height` x `width`.
    """
    # Temporal sampling.
    clip_seq = []
    for s, e in zip(start_idx, end_idx):
        clip_seq.append(
            temporal_sampling(
                frames,
                s,
                e,
                video_num_frames,
            )
        )
    clip_seq = torch.stack(clip_seq, dim=0)

    # Convert frames of the uint type in the range [0, 255] to
    # a torch.FloatTensor in the range [0.0, 1.0]
    clip_seq = clip_seq.float()
    clip_seq = clip_seq / 255.0

    # S T H W C -> SxT C H W
    clip_seq = clip_seq.view(
        clip_seq.shape[0] * clip_seq.shape[1],
        clip_seq.shape[2],
        clip_seq.shape[3],
        clip_seq.shape[4],
    )
    clip_seq = clip_seq.permute(0, 3, 1, 2)
    # Visual transformations.
    clip_seq = apply_visual_transform(
        clip_seq,
        spatial_idx=spatial_sample_index,
        crop_size=crop_size,
    )
    # SxT C H W -> S C T H W
    clip_seq = clip_seq.reshape(
        len(start_idx),
        video_num_frames,
        clip_seq.shape[1],
        clip_seq.shape[2],
        clip_seq.shape[3],
    )
    clip_seq = clip_seq.transpose(1, 2).contiguous()

    return clip_seq


def apply_visual_transform(
    frames,
    spatial_idx,
    crop_size=128,
):
    """
    Apply visual data transformations on the given video frames.
    Data transformations include `resize_crop`, `flip` and `color_normalize`.
    Args:
        cfg (CfgNode): configs.
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `channel` x `height` x `width`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [0, 1, 2]
    # The testing is deterministic and no jitter should be performed.
    # min_scale, max_scale, and crop_size are expect to be the same.

    frames = short_side_scale_jitter(frames, crop_size)
    frames = uniform_crop(frames, crop_size, spatial_idx)
    frames = color_normalization(frames, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
    return frames


def short_side_scale_jitter(images, crop_size):
    """
    Perform a spatial short scale jittering on the given images.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    size = crop_size

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (height <= width and height == size):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="nearest",
        # align_corners=False,
    )


def uniform_crop(images, size, spatial_idx):
    """
    Perform uniform spatial sampling on the images.
    args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
    returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    return cropped


def color_normalization(images, mean, stddev):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1], "channel mean not computed properly"
    assert len(stddev) == images.shape[1], "channel stddev not computed properly"

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]

    return out_images


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_audio_seq(waveform, start_idx, end_idx, audio_fps, frequency, time):
    """
    Sample a sequence of clips from the input audio, and apply
    audio transformations.
    args:
        waveform (tensor): a tensor of audio waveform, dimension is
            `channel` x `time`.
        start_idx (list): list of the start index.
        end_idx (list): list of the end index (not inclusive).
        apply_transform (bool): whether to apply transformations.
    returns:
        (tensor): a sequence of log-mel-scaled spectrogram with dimension of
            `sequence length` x `channel` x `frequency` x `time`.
    """
    audio_seq = []
    for s, e in zip(start_idx, end_idx):
        # Temporal sampling.
        waveform_view = waveform[:, s:e]
        # Convert it to log-mel-scaled spectrogram.
        log_mel_spectrogram = get_log_mel_spectrogram(
            waveform_view, audio_fps, frequency, time
        )
        audio_seq.append(log_mel_spectrogram)
    # S x C x F x T
    audio_seq = torch.stack(audio_seq, dim=0)

    return audio_seq


def resample(waveform, orig_freq, new_freq, use_mono=True):
    """
    Resample the input waveform to ``new_freq``.
    args:
        waveform (tensor): waveform to perform resampling. The dimension is
            `channel` x `frequency` x `width`.
        `orig_freq` (int): original sampling rate of `waveform`.
        `new_freq` (int): target sampling rate of `waveform`.
        `use_mono` (bool): If True, first convert `waveform` to a monophonic signal.
    returns:
         (tensor): waveform with dimension of
            `channel` x `time`.
    """
    if waveform.size(0) != 1 and use_mono:
        waveform = waveform.mean(0, keepdim=True)

    if orig_freq != new_freq:
        waveform = torchaudio.transforms.Resample(
            orig_freq,
            new_freq,
        )(waveform)

    return waveform


def get_log_mel_spectrogram(
    waveform,
    audio_fps,
    frequency,
    time,
):
    """
    Convert the input waveform to log-mel-scaled spectrogram.
    args:
        waveform (tensor): input waveform. The dimension is
            `channel` x `time.`
        `audio_fps` (int): sampling rate of `waveform`.
        `frequency` (int): target frequecy dimension (number of mel bins).
        `time` (int): target time dimension.
    returns:
        (tensor): log-mel-scaled spectrogram with dimension of
            `channel` x `frequency` x `time`.
    """
    w = waveform.size(-1)
    n_fft = 2 * (math.floor(w / time) + 1)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        audio_fps,
        n_fft=n_fft,
        n_mels=frequency,
    )(waveform)
    log_mel_spectrogram = torch.log(1e-6 + mel_spectrogram)
    _nchannels, _frequency, _time = log_mel_spectrogram.size()
    assert _frequency == frequency, f"frequency {_frequency} must be {frequency}"
    if _time != time:
        t = torch.zeros(
            _nchannels,
            frequency,
            time,
            dtype=log_mel_spectrogram.dtype,
        )
        min_time = min(time, _time)
        t[:, :, :min_time] = log_mel_spectrogram[:, :, :min_time]
        log_mel_spectrogram = t

    return log_mel_spectrogram
