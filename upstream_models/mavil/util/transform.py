#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math

# import cv2
import random
import re

import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms

# from .rand_augment import rand_augment_transform RandomErasing


_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def random_short_side_scale_jitter(
    images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (ndarray): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale, max_scale].
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    if inverse_uniform_sampling:
        size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (height <= width and height == size):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
    )


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_crop(images, size, boxes=None):
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]

    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None

    return cropped, cropped_boxes


def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        flipped_boxes (ndarray or None): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

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
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


def clip_boxes_to_image(boxes, height, width):
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (ndarray): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (ndarray): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = 0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    return out_images


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
    if len(images.shape) == 3:
        assert len(mean) == images.shape[0], "channel mean not computed properly"
        assert len(stddev) == images.shape[0], "channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert len(mean) == images.shape[1], "channel mean not computed properly"
        assert len(stddev) == images.shape[1], "channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = (images[idx] - mean[idx]) / stddev[idx]
        elif len(images.shape) == 4:
            out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
        else:
            raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")
    return out_images


def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    This is similar to random_resized_crop. However, it samples two different
    boxes (for cropping) for the first and last frame. It then linearly
    interpolates the two boxes for other frames.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out


def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """
    Get video randaug transform.

    Args:
        input_size: The size of the input video in tuple.
        auto_augment: Parameters for randaug. An example:
            "rand-m7-n4-mstd0.5-inc1" (m is the magnitude and n is the number
            of operations to apply).
        interpolation: Interpolation method.
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return transforms.Compose([rand_augment_transform(auto_augment, aa_params)])
    raise NotImplementedError


def random_sized_crop_img(
    im,
    size,
    jitter_scale=(0.08, 1.0),
    jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
    max_iter=10,
):
    """
    Performs Inception-style cropping (used for training).
    """
    assert len(im.shape) == 3, "Currently only support image for random_sized_crop"
    h, w = im.shape[1:3]
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    cropped = im[:, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


# The following code are modified based on timm lib, we will replace the following
# contents with dependency from PyTorchVideo.
# https://github.com/facebookresearch/pytorchvideo
class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(tuple(round(s, 4) for s in self.scale))
        format_string += ", ratio={0}".format(tuple(round(r, 4) for r in self.ratio))
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0

_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": _FILL,
}

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0],
            -rotn_center[1] - post_trans[1],
            matrix,
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    """
    Apply for video.
    """

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = {
            "fillcolor": hparams["img_mean"] if "img_mean" in hparams else _FILL,
            "resample": hparams["interpolation"]
            if "interpolation" in hparams
            else _RANDOM_INTERPOLATION,
        }

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img_list):
        if self.prob < 1.0 and random.random() > self.prob:
            return img_list
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = (
            self.level_fn(magnitude, self.hparams) if self.level_fn is not None else ()
        )

        if isinstance(img_list, list):
            return [self.aug_fn(img, *level_args, **self.kwargs) for img in img_list]
        else:
            return self.aug_fn(img_list, *level_args, **self.kwargs)


_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
        for name in transforms
    ]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams):
    """
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719

    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :return: A PyTorch compatible Transform
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split("-")
    assert config[0] == "rand"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert NotImplementedError
    ra_ops = rand_augment_ops(
        magnitude=magnitude, hparams=hparams, transforms=transforms
    )
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device="cuda",
        cube=True,
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.cube = cube
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(100):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    for i in range(batch_start, batch_size):
                        img_instance = img[i]
                        img_instance[:, top : top + h, left : left + w] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=self.device,
                        )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            if self.cube:
                self._erase_cube(
                    input,
                    batch_start,
                    batch_size,
                    chan,
                    img_h,
                    img_w,
                    input.dtype,
                )
            else:
                for i in range(batch_start, batch_size):
                    self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
