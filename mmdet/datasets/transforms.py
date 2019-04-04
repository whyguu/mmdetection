import mmcv
import numpy as np
import torch
import glob
import cv2

__all__ = ['ImageTransform', 'BboxTransform', 'MaskTransform', 'Numpy2Tensor']


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        # tricks using normal img
        self.normal_img_prefix = 'x-ray/jinnan2_round1_train_20190305/normal/'
        self.normal_img = glob.glob1(self.normal_img_prefix, '*jpg')

    def __call__(self, img, scale, flip=False, keep_ratio=True,
                 flip_ud=False, transpose=False, use_norm_img=False):
        if use_norm_img:
            if np.random.rand() < 0.5:
                norm_img_idx = np.random.randint(0, len(self.normal_img))
                normal_img = mmcv.imread(self.normal_img_prefix+self.normal_img[norm_img_idx])
                h, w = img.shape[0:2]
                h1, w1 = normal_img.shape[0:2]
                if (h > w) != (h1 > w1):
                    normal_img = normal_img.transpose(1, 0, 2)
                normal_img = mmcv.imrescale(normal_img, (h, w), return_scale=False)
                h1, w1 = normal_img.shape[0:2]
                # norm img weight
                # alpha = np.random.uniform(0.1, 0.9)
                alpha = 0.5
                cv2.addWeighted(normal_img, alpha, img[0:h1, 0:w1], 1-alpha, 0, img[0:h1, 0:w1])
                # cv2.imwrite('mix_norm.jpg', img)
                # exit()

        if transpose:
            img = img.transpose(1, 0, 2)
            sl = list(scale)
            sl.reverse()
            scale = tuple(sl)
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if flip_ud:
            img = mmcv.imflip(img, direction='vertical')
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_flip_ud(bboxes, img_shape):
    """Flip bboxes vertically.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    h = img_shape[0]
    flipped = bboxes.copy()
    flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
    flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
    return flipped


def bbox_transpose(bboxes, img_shape=None):
    """transpose bboxes (not equals rot 90 anti-clock).

    Args:
        bboxes(ndarray): shape (..., 4*k)
        # img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.copy()
    xs = flipped[..., 0::2].copy()
    flipped[..., 0::2] = flipped[..., 1::2]
    flipped[..., 1::2] = xs

    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False, flip_ud=False, transpose=False):
        gt_bboxes = bboxes
        if transpose:
            gt_bboxes = bbox_transpose(gt_bboxes, img_shape)
        gt_bboxes = gt_bboxes * scale_factor
        if flip_ud:
            gt_bboxes = bbox_flip_ud(gt_bboxes, img_shape)

        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
