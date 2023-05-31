import os
import argparse
from datetime import datetime
import torch
from pathlib import Path
"""Configs."""
from fvcore.common.config import CfgNode
import warnings

project_dir = str(Path(__file__).resolve().parent.parent)
dataset_root = os.path.join(project_dir, 'datasets')
output_root = os.path.join(project_dir, 'runs')


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------------- #
# Pretraining options.
# ---------------------------------------------------------------------------- #
_C.PRETRAIN = CfgNode()

# Path to the checkpoint to load the pretrained weight.
_C.PRETRAIN.CHECKPOINT_FILE_PATH = ""


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "UCF101"

# Dataset split.
_C.TRAIN.DATASET_SPLIT = 1

# Number of samples to sample from a data file.
_C.TRAIN.NUM_SAMPLES = 10

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 32

# Evaluate model on validation data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Evaluate model on test data every eval period epochs.
_C.TRAIN.TEST_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.SAVE_EVERY_EPOCH = 1

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# PREEMPTIBLE
_C.TRAIN.PREEMPTIBLE = False


# ---------------------------------------------------------------------------- #
# Validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()

# If True validate the model, else skip the testing.
_C.VAL.ENABLE = False

# Dataset for validation.
_C.VAL.DATASET = "UCF101"

# Dataset split.
_C.VAL.DATASET_SPLIT = 1


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "UCF101"

# Dataset split.
_C.TEST.DATASET_SPLIT = 1

# Total mini-batch size
_C.TEST.BATCH_SIZE = 16

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of samples to sample from a data file.
_C.TEST.NUM_SAMPLES = 30

# Number of samples to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3


# -----------------------------------------------------------------------------
# Visual options
# -----------------------------------------------------------------------------
_C.VIS = CfgNode()

# Visual Model architecture.
_C.VIS.ARCH = "resnet"

# Visual Model name.
_C.VIS.MODEL_NAME = "ResNet"


# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Audio options
# -----------------------------------------------------------------------------
_C.AUD = CfgNode()

# Audio Model architecture.
_C.AUD.ARCH = "resnet"

# Audio Model name
_C.AUD.MODEL_NAME = "AudioResNet"


# -----------------------------------------------------------------------------
# AudioResNet options
# -----------------------------------------------------------------------------
_C.AUDIO_RESNET = CfgNode()

# Transformation function.
_C.AUDIO_RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.AUDIO_RESNET.NUM_GROUPS = 1

# Width of each group (32 -> ResNet; 4 -> ResNeXt).
_C.AUDIO_RESNET.WIDTH_PER_GROUP = 32

# Apply relu in a inplace manner.
_C.AUDIO_RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.AUDIO_RESNET.STRIDE_1X1 = False

# Number of weight layers.
_C.AUDIO_RESNET.DEPTH = 50

# Size of stride on different res stages.
_C.AUDIO_RESNET.STRIDES = [2, 2, 2, 2]

# Size of dilation on different res stages.
_C.AUDIO_RESNET.DILATIONS = [1, 1, 1, 1]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------

_C.MODEL = CfgNode()

# Downstream task.
_C.MODEL.TASK = "VisualClassify"

# Modality fusion strategy for downstream task.
_C.MODEL.DOWNSTREAM_FUSION = "late"

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

#  If true, initialize the gamma of the final BN of each block of ResNet to zero.
_C.MODEL.ZERO_INIT_FINAL_BN = True

# Epsilon value for normalization layers.
_C.MODEL.EPSILON = 1e-5

# Momentum value for normalization layers.
_C.MODEL.MOMENTUM = 0.1

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 101

# Dropout rate.
_C.MODEL.DROPOUT_RATE = 0.5

# If True, average-pool feature maps from convolutional models and feed
# them into another models (e.g. Transformers).
# If False, average-pool feature maps from convolutional models and directly
# perform prediction.
_C.MODEL.POOLING = False

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Margin value for MultiMarginLoss.
_C.MODEL.MARGIN = 0.1

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Transformer architecture.
_C.MODEL.ARCH = "visual_audio"

# Transformer modality fusion strategy.
_C.MODEL.FUSION = "mid"

# Whether to use Transformers.
_C.MODEL.USE_TRANSFORMER = False


# -----------------------------------------------------------------------------
# Transformer options
# -----------------------------------------------------------------------------

_C.TRANSFORMER = CfgNode()

# The ropout ratio for the attention probabilities
_C.TRANSFORMER.ATTENTION_PROBS_DROPOUT_PROB = 0.1

# The non-linear activation function in the encoder.
_C.TRANSFORMER.HIDDEN_ACT = "gelu"

# The dropout probability for all fully connected layers in the encoder.
_C.TRANSFORMER.HIDDEN_DROPOUT_PROB = 0.1

# Size of the encoder layers.
_C.TRANSFORMER.HIDDEN_SIZE = 768

# The stddev of the normal initializer for initializing weight matrices.
_C.TRANSFORMER.INITIALIZER_RANGE = 0.02

# The size of the "intermediate" (i.e., feed-forward) layer in the encoder.
_C.TRANSFORMER.INTERMEDIATE_SIZE = 3072

# The maximum sequence length
_C.TRANSFORMER.MAX_POSITION_EMBEDDINGS = 22

# Number of attention heads for each attention layer in the encoder.
_C.TRANSFORMER.NUM_ATTENTION_HEADS = 12

# Number of hidden layers in the encoder.
_C.TRANSFORMER.NUM_HIDDEN_LAYERS = 6

# Number of group for the hidden layers, parameters in the same group are shared.
_C.TRANSFORMER.NUM_HIDDEN_GROUPS = 1

# Network structure type.
_C.TRANSFORMER.NET_STRUCTURE_TYPE = 0

# Gap size.
_C.TRANSFORMER.GAP_SIZE = 0

# Number of memory blocks.
_C.TRANSFORMER.NUM_MEMORY_BLOCKS = 0

# Number of inner repetition of attention and ffn.
_C.TRANSFORMER.INNER_GROUP_NUM = 1

# The scale to apply.
_C.TRANSFORMER.DOWN_SCALE_FACTOR = 1

# The vocabular size (number of modalities) of the `token_type_ids`
_C.TRANSFORMER.TYPE_VOCAB_SIZE = 2

# Whether to output hidden states of all layers or not.
_C.TRANSFORMER.OUTPUT_HIDDEN_STATES = False

# Whether to use mean pooling for positional encoding.
_C.TRANSFORMER.USE_MEAN_POOLING = False

# Parameter sharing across Transformer layers.
_C.TRANSFORMER.SHARING_ACROSS_LAYERS = True

# Parameter sharing across modalities.
_C.TRANSFORMER.SHARING_ACROSS_MODELS = True

# Perform low-rank decomposition.
_C.TRANSFORMER.USE_DECOMPOSITION = True

# Low-rank decomposition size
_C.TRANSFORMER.ORTHOGONAL_SIZE = 128

# Method to impose the orthogonalization constraint.
_C.TRANSFORMER.ORTHOGONAL_MODE = "static"

# Method to initialize orthogonal matrices.
_C.TRANSFORMER.ORTHOGONAL_INIT = "cayley"

# Hyperparameter for imposing the orthogonalization constraint.
_C.TRANSFORMER.ORTHOGONAL_K = 100


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 112

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 32

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 2

# Input videos may have different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3]

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial crop size for training
_C.DATA.TRAIN_CROP_SIZE = 112

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [128, 160]

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 128

# Input audio clip duration (sec)
_C.DATA.CLIP_DURATION = 2

# Input audios may have different sampling rate, convert it to the target audio
# sampling rate.
_C.DATA.TARGET_AUDIO_RATE = 44100

# Number of mel bins for log-mel-scaled spectrograms.
_C.DATA.AUDIO_FREQUENCY = 80

# Time dimension for log-mel-scaled spectrograms.
_C.DATA.AUDIO_TIME = 128

# The audio frequency masking droput rate.
_C.DATA.FREQUENCY_MASK_RATE = 0.05

# The audio temporal masking dropout rate.
_C.DATA.TIME_MASK_RATE = 0.05

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# List of data augmentations.
_C.DATA.TRANSFORMATIONS = ["resize_crop", "flip", "color_normalize"]

# Sequene length of Transformer inputs.
_C.DATA.SEQUENCE_LENGTH = 10


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.03

# Base finetuning learning rate
_C.SOLVER.FINETUNE_LR = 0.0001

# Evaluation protocol, options include "linear_eval" and "finetune".
_C.SOLVER.PROTOCOL = "finetune"

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 10

# Maximal number of steps.
_C.SOLVER.NUM_STEPS = -1

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Use AMSGrad
_C.SOLVER.USE_AMSGRAD = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 5e-6

# L2 regularization for finetuning.
_C.SOLVER.FINETUNE_WEIGHT_DECAY = 5e-6

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# The start finetuning learning rate of the warm up.
_C.SOLVER.WARMUP_START_FINETUNE_LR = 0.01

# Gradually warm up the lrs over this number of steps.
_C.SOLVER.WARMUP_STEPS = -1

# Gradually warm up the lrs over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# Gradually warm up the lrs over the first proportion of total steps.
_C.SOLVER.WARMUP_PROPORTION = 0.0

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adam"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use per machine (applies to both training and testing).
_C.NUM_GPUS = -1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Data basedir.
_C.DATASET_ROOT = ""

# Data dir.
_C.DATASET_DIR = ""

# Output basedir.
_C.OUTPUT_ROOT = ""

# Checkpoints dir.
_C.SAVE_DIR = ""

# Log dir.
_C.LOG_DIR = ""

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = -1

# Log period in iters.
_C.LOG_PERIOD = 50

# Distributed init method.
_C.DIST_INIT_METHOD = "tcp://localhost:9999"

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 16

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


def get_video_cfg():
    """
    Get a copy of the default config.
    """
    config = _C.clone()
    config.MODEL.TASK = "VisualClassify"
    config.MODEL.DOWNSTREAM_FUSION = "late"
    config.MODEL.DROPOUT_RATE = 0.5
    config.MODEL.POOLING = False
    config.MODEL.USE_TRANSFORMER = False

    return config

def get_audio_cfg():
    """
    Get a copy of the default config.
    """
    config = _C.clone()
    config.MODEL.TASK = "AudioClassify"
    config.MODEL.DOWNSTREAM_FUSION = "late"
    config.MODEL.DROPOUT_RATE = 0.1
    config.MODEL.POOLING = True
    config.MODEL.USE_TRANSFORMER = False

    return config

def get_multi_cfg():
    """
    Get a copy of the default config.
    """
    config = _C.clone()
    config.MODEL.TASK = "MultimodalSequenceClassify"
    config.MODEL.DOWNSTREAM_FUSION = "concat"
    config.MODEL.DROPOUT_RATE = 0.5
    config.MODEL.POOLING = True
    config.MODEL.USE_TRANSFORMER = True

    return config
