import torch
import torchvision
from pathlib import Path

import utils.logging as logging
from data.build import DATASET_REGISTRY
import data.transform as transform
import data.utils as utils


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class KineticsSounds(torch.utils.data.Dataset):
    """
    Kinetics-Sounds video loader. Construct the Kinetics-Sounds video loader,
    then sample audio/visual clips from the videos. For training and validation,
    multiple audio/visual clips are uniformly sampled from every video with
    audio/visual random transformations. For testing, multiple audio/visual
    clips are uniformaly sampled from every video with only uniform cropping.
    For uniform cropping, we take the left, center, and right crop
    if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Mode {} not suported for KineticsSounds".format(mode)

        self.mode = mode
        self.cfg = cfg

        if self.mode in ['train', 'val']:
            self._num_clips = cfg.TRAIN.NUM_SAMPLES
        elif self.mode in ['test']:
            self._num_clips = (
                cfg.TEST.NUM_SAMPLES
            )
            assert cfg.TEST.NUM_SAMPLES == cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS, \
                f"test num samples {cfg.TEST.NUM_SAMPLES} must be #views {cfg.TEST.NUM_ENSEMBLE_VIEWS} x #crops {cfg.TEST.NUM_SPATIAL_CROPS}"

        logger.info(f"Constructin KineticsSounds mode {self.mode}")
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode = self.mode
        data_dir = Path(self.cfg.DATASET_DIR).joinpath(f"{mode}")

        self.idx2label = sorted(data_dir.iterdir())
        self.idx2label = [path.name for path in self.idx2label]
        self.label2idx = {label: idx for idx, label in enumerate(self.idx2label)}

        videos = sorted(data_dir.rglob("*.mp4"))

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        for video_idx, video_path in enumerate(videos):
            yid = video_path.stem
            path = str(video_path)
            label = self.label2idx[video_path.parent.name]
            if mode in ["train", "val"]:
                self._path_to_videos.append(path)
                self._labels.append(label)
                self._spatial_temporal_idx.append(list(range(self._num_clips)))
            elif mode in ["test"]:
                for idx in range(self.cfg.TEST.NUM_SPATIAL_CROPS):
                    self._path_to_videos.append(path)
                    self._labels.append(label)
                    self._spatial_temporal_idx.append(
                        [
                            idx * self.cfg.TEST.NUM_ENSEMBLE_VIEWS + i
                            for i in range(self.cfg.TEST.NUM_ENSEMBLE_VIEWS)
                        ]
                    )

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load KineticsSounds mode {}".format(
            self.mode,
        )
        logger.info(
            "Constructing KineticsSounds dataloader (mode: {}, size: {})".format(
                self.mode, len(self._path_to_videos),
            )
        )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def __getitem__(self, index):
        """
        Given the video index, return the audio/visual sequence of clips,
        the list of labels and the list of video index.
        video index.
        args:
            index (int): the video index provided by the pytorch sampler.
        returns:
            visual_seq (tensor): a sequence of sampled visual clips.
                `sequence length` x `channel` x `num frames` x `height` x `width`.
            audio_seq (tensor): a sequence of log-mel-scaled spectrograms.
                `sequence length` x `channel` x `frequency` x `time`.
            (tensor): list of the label of the current video.
            (tensor): list of the index of the video.
        """
        frames, waveform, info = torchvision.io.read_video(
            self._path_to_videos[index],
            pts_unit="sec",
        )

        video_fps = round(info["video_fps"])
        audio_fps = info["audio_fps"]
        if self.mode in ['train', 'val']:
            temporal_sample_index = self._spatial_temporal_idx[index]
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            num_samples = self._num_clips
        elif self.mode in ['test']:
            offset = self._spatial_temporal_idx[index][0]
            temporal_sample_index = [
                i - offset for i in self._spatial_temporal_idx[index]
            ]
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index][0]
                // self.cfg.TEST.NUM_ENSEMBLE_VIEWS
            )

            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            num_samples = self.cfg.TEST.NUM_ENSEMBLE_VIEWS
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Adjust number of frames consdiering input video fps, taget fps and
        # frame sampling rate.
        _num_frames = (
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE *
            video_fps /
            self.cfg.DATA.TARGET_FPS
        )
        # Compute audio waveform corresponding to the visual clip.
        waveform_size = int(
            self.cfg.DATA.TARGET_AUDIO_RATE *
            self.cfg.DATA.NUM_FRAMES *
            self.cfg.DATA.SAMPLING_RATE /
            self.cfg.DATA.TARGET_FPS
        )
        visual_delta = max(frames.size(0) - _num_frames, 0)
        visual_start_idx = [
            visual_delta * i / (num_samples - 1)
            for i in temporal_sample_index
        ]
        visual_end_idx = [s + _num_frames - 1 for s in visual_start_idx]

        label = self._labels[index]
        visual_seq = self.get_visual_seq(
            frames,
            visual_start_idx,
            visual_end_idx,
            min_scale,
            max_scale,
            crop_size,
            spatial_sample_index,
        )
        waveform = transform.resample(
            waveform,
            audio_fps,
            self.cfg.DATA.TARGET_AUDIO_RATE,
            use_mono=True,
        )
        audio_delta = max(waveform.size(-1) - waveform_size, 0)
        audio_start_idx = [
            int(audio_delta * (idx / visual_delta))
            for idx in visual_start_idx
        ]
        audio_end_idx = [s + waveform_size for s in audio_start_idx]
        audio_seq = self.get_audio_seq(
            waveform,
            audio_start_idx,
            audio_end_idx,
            True if self.mode in ['train'] else False,
        )
        _label = torch.LongTensor([label] * num_samples)
        _index = torch.LongTensor(
            [index * num_samples + i for i in range(num_samples)]
        )
        return visual_seq, audio_seq, _label, _index

    def get_visual_seq(
        self,
        frames,
        start_idx,
        end_idx,
        min_scale,
        max_scale,
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
                utils.temporal_sampling(
                    frames,
                    s,
                    e,
                    self.cfg.DATA.NUM_FRAMES,
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
        clip_seq = utils.apply_visual_transform(
            self.cfg,
            clip_seq,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )
        # SxT C H W -> S C T H W
        clip_seq = clip_seq.reshape(
            len(start_idx),
            self.cfg.DATA.NUM_FRAMES,
            clip_seq.shape[1],
            clip_seq.shape[2],
            clip_seq.shape[3],
        )
        clip_seq = clip_seq.transpose(1, 2).contiguous()
        clip_seq = [clip_seq]

        return clip_seq

    def get_audio_seq(
        self,
        waveform,
        start_idx,
        end_idx,
        apply_transform=False,
    ):
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
            log_mel_spectrogram = transform.get_log_mel_spectrogram(
                waveform_view,
                self.cfg.DATA.TARGET_AUDIO_RATE,
                self.cfg.DATA.AUDIO_FREQUENCY,
                self.cfg.DATA.AUDIO_TIME,
            )
            audio_seq.append(log_mel_spectrogram)
        # S x C x F x T
        audio_seq = torch.stack(audio_seq, dim=0)

        # Apply transformations.
        if apply_transform:
            audio_seq = utils.apply_audio_transform(
                audio_seq,
                self.cfg.DATA.FREQUENCY_MASK_RATE,
                self.cfg.DATA.TIME_MASK_RATE,
            )
        return audio_seq
