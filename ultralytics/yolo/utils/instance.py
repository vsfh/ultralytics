# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """From PyTorch internals."""

    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(yolo format)
# `ltwh` means left top and width, height(coco format)
_formats = ['xyxy', 'xywh', 'ltwh']

__all__ = 'Bboxes',  # tuple or list


class Bboxes:
    """Now only numpy is supported."""

    def __init__(self, bboxes, format='xyxy') -> None:
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    # def convert(self, format):
    #     assert format in _formats
    #     if self.format == format:
    #         bboxes = self.bboxes
    #     elif self.format == "xyxy":
    #         if format == "xywh":
    #             bboxes = xyxy2xywh(self.bboxes)
    #         else:
    #             bboxes = xyxy2ltwh(self.bboxes)
    #     elif self.format == "xywh":
    #         if format == "xyxy":
    #             bboxes = xywh2xyxy(self.bboxes)
    #         else:
    #             bboxes = xywh2ltwh(self.bboxes)
    #     else:
    #         if format == "xyxy":
    #             bboxes = ltwh2xyxy(self.bboxes)
    #         else:
    #             bboxes = ltwh2xywh(self.bboxes)
    #
    #     return Bboxes(bboxes, format)

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        if self.format == format:
            return
        elif self.format == 'xyxy':
            bboxes = xyxy2xywh(self.bboxes) if format == 'xywh' else xyxy2ltwh(self.bboxes)
        elif self.format == 'xywh':
            bboxes = xywh2xyxy(self.bboxes) if format == 'xyxy' else xywh2ltwh(self.bboxes)
        else:
            bboxes = ltwh2xyxy(self.bboxes) if format == 'xyxy' else ltwh2xywh(self.bboxes)
        self.bboxes = bboxes
        self.format = format

    def areas(self):
        """Return box areas."""
        self.convert('xyxy')
        return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])

    def rot_90(self, M, angle):
        n = len(self.bboxes)
        new_bbox = np.zeros_like(self.bboxes)
        x1y1 = np.ones((n,3))
        x2y2 = np.ones((n,3))
        
        if angle == -90:
            x1y1[:,:2] = self.bboxes[:,[0,3]]
            x2y2[:,:2] = self.bboxes[:,[2,1]]

        elif angle == -180:
            x1y1[:,:2] = self.bboxes[:,[2,3]]
            x2y2[:,:2] = self.bboxes[:,[0,1]]
        
        elif angle == 90:
            x1y1[:,:2] = self.bboxes[:,[2,1]]
            x2y2[:,:2] = self.bboxes[:,[0,3]]    

        elif angle == 180:
            x1y1[:,:2] = self.bboxes[:,[2,3]]
            x2y2[:,:2] = self.bboxes[:,[0,1]]
        
        elif angle == 0:
            return
        else:
            print('error rot angle')
        
        new_bbox[:,:2] = (x1y1 @ M.T)[:,:2]
        new_bbox[:,-2:] = (x2y2 @ M.T)[:,:2]
        
        self.bboxes = new_bbox
        
    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Args:
            scale (tuple) or (list) or (int): the scale for four coords.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Args:
            offset (tuple) or (list) or (int): the offset for four coords.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)
    
    @classmethod
    def concatenate(cls, boxes_list: List['Bboxes'], axis=0) -> 'Bboxes':
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.

        Returns:
            Bboxes: A new Bboxes object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> 'Bboxes':
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].view(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f'Indexing on Bboxes with {index} failed to return a matrix!'
        return Bboxes(b)


class Instances:

    def __init__(self, bboxes, pose, segments=None, keypoints=None, bbox_format='xywh', normalized=True) -> None:
        """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
        if segments is None:
            segments = []
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.pose = pose
        self.normalized = normalized

        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            segments = resample_segments(segments)
            # (N, 1000, 2)
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        self.segments = segments

    def convert_bbox(self, format):
        """Convert bounding box format."""
        self._bboxes.convert(format=format)
    
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        self._bboxes.areas()

    def rot_90_bbox(self, M, angle):
        self.pose[:, 2] += -angle
        self._bboxes.rot_90(M, angle)
        
    def scale(self, scale_w, scale_h, bbox_only=False):
        """this might be similar with denormalize func but without normalized sign."""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        # self.segments[..., 0] *= w
        # self.segments[..., 1] *= h
        # if self.keypoints is not None:
        #     self.keypoints[..., 0] *= w
        #     self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """Normalize bounding boxes, segments, and keypoints to image dimensions."""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation."""
        assert not self.normalized, 'you should add padding with absolute coordinates.'
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> 'Instances':
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        pose = self.pose[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            pose=pose,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        """Flips the coordinates of bounding boxes, segments, and keypoints vertically."""
        if self._bboxes.format == 'xyxy':
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w):
        """Reverses the order of the bounding boxes and segments horizontally."""
        if self._bboxes.format == 'xyxy':
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.pose[0,0] = -self.pose[0,0]
        self.pose[0,1] = -self.pose[0,1]
        # self.segments[..., 0] = w - self.segments[..., 0]
        # if self.keypoints is not None:
        #     self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        """Clips bounding boxes, segments, and keypoints values to stay within image boundaries."""
        ori_format = self._bboxes.format
        self.convert_bbox(format='xyxy')
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != 'xyxy':
            self.convert_bbox(format=ori_format)
        self.segments[..., 0] = self.segments[..., 0].clip(0, w)
        self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def update(self, bboxes, segments=None, keypoints=None):
        """Updates instance variables."""
        new_bboxes = Bboxes(bboxes, format=self._bboxes.format)
        self._bboxes = new_bboxes
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """Return the length of the instance list."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List['Instances'], axis=0) -> 'Instances':
        """
        Concatenates a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_poses = np.concatenate([ins.pose for ins in instances_list], axis=axis)
        
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_poses, bbox_format=bbox_format, normalized=normalized)

    @property
    def bboxes(self):
        """Return bounding boxes."""
        return self._bboxes.bboxes
