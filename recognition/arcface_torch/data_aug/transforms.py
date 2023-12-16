import math
import random
from abc import ABC

import cv2
import time
import numpy as np
from PIL import Image
from albumentations.core.transforms_interface import (ImageOnlyTransform, to_tuple)
from insightface.app import MaskAugmentation
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn


class CutOut(ImageOnlyTransform):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 always_apply=False,
                 p=1.0,
                 ):
        super(CutOut, self).__init__(always_apply, p)

        assert isinstance(n_holes, int) and n_holes > 0, 'n_holes should be int and great than 0'
        if cutout_ratio is None:
            cutout_ratio = [0.05, 0.1, 0.2]
        self.cut_ratio = cutout_ratio
        self.n_holes = n_holes
        self.fill_in = fill_in

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if ((random.random() < self.p) or self.always_apply or force_apply) and (not kwargs.get('mouth_muffle', False)):
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                                                    " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)
        kwargs['Cutout'] = False

        return kwargs

    def apply(self, img, **params):
        # """Call function to drop some regions of image."""
        # start_cutout = time.time()
        h, w, c = img.shape
        n_holes = np.random.randint(1, self.n_holes)
        for _ in range(n_holes):
            w_ratio = np.random.choice(self.cut_ratio)
            h_ratio = np.random.choice(self.cut_ratio)
            w_holes = w * w_ratio
            h_holes = h * h_ratio
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)

            x2 = int(np.clip(x1 + w_holes, 0, w))
            y2 = int(np.clip(y1 + h_holes, 0, h))

            img[y1:y2, x1:x2, :] = self.fill_in
        # print(f'cutout consume time: {time.time() - start_cutout}')
        return img

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        res = super(CutOut, self).apply_with_params(params, **kwargs)
        res['Cutout'] = True
        res['mouth_muffle'] = kwargs['mouth_muffle']
        # print('mask augmentation')
        return res

    @property
    def targets_as_params(self):
        return ["image", 'mouth_muffle']

    def get_params_dependent_on_targets(self, params):
        mouth_muffle = params.get('mouth_muffle', False)
        return {'mouth_muffle': mouth_muffle}

    def get_transform_init_args_names(self):
        return ()


class MaskAugmentationAdd(MaskAugmentation):
    def __init__(
            self,
            mask_names=None,
            mask_probs=None,
            h_low=0.33,
            h_high=0.35,
            always_apply=False,
            p=1.0,
    ):
        if mask_probs is None:
            mask_probs = [0.4, 0.4, 0.1, 0.1]
        if mask_names is None:
            mask_names = ['mask_white', 'mask_blue', 'mask_black', 'mask_green']
        super(MaskAugmentationAdd, self).__init__(mask_names, mask_probs, h_low, h_high, always_apply, p)

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                                                    " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        kwargs['mouth_muffle'] = False
        return kwargs

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        res = super(MaskAugmentationAdd, self).apply_with_params(params, **kwargs)
        res['mouth_muffle'] = True
        # print('mask augmentation')
        return res


class GridMask(ImageOnlyTransform):
    def __init__(self, use_h=True, use_w=True, rotate=1, ratio=0.5, mode=0, always_apply=False, p=1.):
        assert 0 < ratio < 1, 'ratio should be 0 < ratio < 1'
        assert use_h or use_w, 'use_h and use_w should specify at least one of them'
        super(GridMask, self).__init__(always_apply, p)
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode

    def apply(self, img, **params):
        if (not params.get('mouth_muffle', False)) and (not params.get('Cutout', False)):
            # start_gridmask = time.time()
            # print('GridMask augmentation')
            h, w, c = img.shape
            hh = int(1.5 * h)
            ww = int(1.5 * w)
            d = np.random.randint(2, h)
            # d = self.d
            # self.l = int(d*self.ratio+0.5)
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

            mask = np.ones((hh, ww), np.float32)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)
            if self.use_h:
                for i in range(0, hh // d, 2):
                    s = d * i + st_h
                    t = min(s + self.l, hh)
                    mask[s:t, :] *= 0
            if self.use_w:
                for i in range(0, ww // d, 2):
                    s = d * i + st_w
                    t = min(s + self.l, ww)
                    mask[:, s:t] *= 0

            r = np.random.randint(self.rotate)
            mask = Image.fromarray(np.uint8(mask))
            # mask = np.uint8(mask)
            mask = mask.rotate(r)
            mask = np.asarray(mask)
            #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
            mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

            # mask = torch.from_numpy(mask).float().cuda()
            if self.mode == 1:
                mask = 1 - mask

            # mask = mask.expand_as(x)
            mask = np.expand_dims(mask, -1)
            mask = np.concatenate((mask, mask, mask), -1)

            img = img * mask
            # print(f'gridmask consume time: {time.time() - start_gridmask}')
        return img

    @property
    def targets_as_params(self):
        return ["image", 'Cutout', 'mouth_muffle']

    def get_params_dependent_on_targets(self, params):
        cut_out = params.get('Cutout', False)
        mouth_muffle = params.get('mouth_muffle', False)
        return {'Cutout': cut_out, 'mouth_muffle': mouth_muffle}

    def get_transform_init_args_names(self):
        return ()


class BlurByBlock(ImageOnlyTransform):
    def __init__(self, h_step=8, w_step=8, blur_limit=1, always_apply=False, p=1.):
        super(BlurByBlock, self).__init__(always_apply, p)
        assert isinstance(h_step, int), 'h_step should be a int'
        assert isinstance(w_step, int), 'w_step should be a int'
        self.h_step = h_step
        self.w_step = w_step
        self.blur_limit = to_tuple(blur_limit, 1)

    def apply(self, img, **params):
        # start_blur = time.time()
        blur_prob = np.random.random()
        h, w, c = img.shape
        for i in range(0, h, self.h_step):
            for j in range(0, w, self.w_step):
                # Create ROI coordinates
                block_h_max = min(h, i + self.h_step)
                block_w_max = max(w, j + self.w_step)
                # Grab ROI with Numpy slicing and blur
                block = img[i:block_h_max, j:block_w_max]
                kernel = params['kernel']
                if blur_prob < 1 / 3:
                    blur = cv2.GaussianBlur(block, (kernel if kernel % 2 == 1 else 3,
                                                    kernel if kernel % 2 == 1 else 3), 0)
                elif 1 / 3 <= blur_prob < 2 / 3:
                    blur = cv2.blur(block, (kernel if kernel % 2 == 1 else 3,
                                            kernel if kernel % 2 == 1 else 3))
                else:
                    blur = cv2.medianBlur(block, kernel if kernel % 2 == 1 else 3)
                # blur = cv2.filter2D(block, ddepth=-1, kernel=1)  # MotionBlur, AdvancedBlur
                # Insert ROI back into image
                img[i:block_h_max, j:block_w_max] = blur
        # print(f'blur consume time: { time.time() - start_blur}')
        return img

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_transform_init_args_names(self):
        return ()

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        return {"kernel": ksize}


class Enlight(ImageOnlyTransform):
    def __init__(self, strength_light=129, always_apply=False, radius_range=None, p=1.):
        if radius_range is None:
            radius_range = [10, -1]
        else:
            assert len(radius_range) == 2, 'radius range length should be 2'
            if radius_range[1] != -1:
                assert radius_range[0] < radius_range[1], 'radius_range[0] should less than radius_range[1]'
        super(Enlight, self).__init__(always_apply, p)
        self.strength_light = strength_light
        self.radius_range = radius_range

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                                                    " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)
        kwargs['enlight'] = False

        return kwargs

    def apply(self, img, **params):
        # start_light = time.time()
        x, y, _ = img.shape
        # radius = 84
        # radius = np.random.randint(10, 32, 1)  #
        min_radius = max(10, self.radius_range[0])
        max_radius = int(min(x, y)) if self.radius_range[1] == -1 else min(int(min(x, y)), self.radius_range[1])
        radius = np.random.randint(min_radius, max_radius, 1)
        # radius = np.random.randint(10, int(min(x, y)), 1)  #
        pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
        pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
        pos_x = int(pos_x[0])
        pos_y = int(pos_y[0])
        radius = int(radius[0])

        for j in range(pos_y - radius, pos_y + radius):
            for i in range(pos_x - radius, pos_x + radius):
                distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
                distance = np.sqrt(distance)
                if distance < radius:
                    result = 1 - distance / radius
                    result = result * self.strength_light
                    # print(result)
                    img[i, j, 0] = min((img[i, j, 0] + result), 255)
                    img[i, j, 1] = min((img[i, j, 1] + result), 255)
                    img[i, j, 2] = min((img[i, j, 2] + result), 255)
        img = img.astype(np.uint8)
        # print(f'light consume time: {time.time() - start_light }')
        return img

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        res = super(Enlight, self).apply_with_params(params, **kwargs)
        res['enlight'] = True
        # print('mask augmentation')
        return res

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_transform_init_args_names(self):
        return ()


class Enlight_v2(Enlight):
    def __init__(self, strength_light=129, always_apply=False, radius_range=None, p=1., weight=0.5):
        super(Enlight_v2, self).__init__(strength_light=strength_light,
                                         always_apply=always_apply, radius_range=radius_range, p=p)
        self.weight = weight

    def apply(self, img, **params):
        height, width, channel = img.shape

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the brightest pixel in the image
        center_loc = (np.random.randint(height), np.random.randint(width))

        min_radius = max(10, self.radius_range[0])
        max_radius = int(min(height, width)) if self.radius_range[1] == -1 else min(int(min(height, width)),
                                                                                    self.radius_range[1])
        radius = np.random.randint(min_radius, max_radius, 1)
        radius = int(radius[0])

        # Create a circle at the location of the brightest pixel
        circle_img = np.zeros_like(gray)
        cv2.circle(circle_img, center_loc, radius, (self.strength_light, self.strength_light, self.strength_light), -1)

        # Apply a Gaussian blur to the circle to create a light source effect
        blur_img = cv2.GaussianBlur(circle_img, (101, 101), 0)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2RGB)
        # Add the blurred circle to the original image
        result = cv2.addWeighted(img, 1, blur_img, self.weight, 0)
        return result


class Shadow(ImageOnlyTransform):
    def __init__(self, strength_light=129, always_apply=False, radius_range=None, p=1.):
        if radius_range is None:
            radius_range = [10, -1]
        else:
            assert len(radius_range) == 2, 'radius range length should be 2'
            radius_range: list[int, int]
            if radius_range[1] != -1:
                assert radius_range[0] < radius_range[1], 'radius_range[0] should less than radius_range[1]'
        super(Shadow, self).__init__(always_apply, p)
        self.strength_light = strength_light
        self.radius_range = radius_range

    def apply(self, img, **params):
        if not params.get('enlight', False):
            # start_shadow = time.time()
            x, y, _ = img.shape
            min_radius = max(10, self.radius_range[0])
            max_radius = int(min(x, y)) if self.radius_range[1] == -1 else min(int(min(x, y)), self.radius_range[1])
            radius = np.random.randint(min_radius, max_radius, 1)
            # radius = np.random.randint(10, 32, 1)  #
            # radius = np.random.randint(10, int(min(x, y)), 1)  #
            pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
            pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
            pos_x = int(pos_x[0])
            pos_y = int(pos_y[0])
            radius = int(radius[0])
            for j in range(pos_y - radius, pos_y + radius):
                for i in range(pos_x - radius, pos_x + radius):
                    distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
                    distance = np.sqrt(distance)
                    if distance < radius:
                        result = 1 - distance / radius
                        result = result * self.strength_light
                        # print(result)
                        img[i, j, 0] = max((img[i, j, 0] - result), 0)
                        img[i, j, 1] = max((img[i, j, 1] - result), 0)
                        img[i, j, 2] = max((img[i, j, 2] - result), 0)
            img = img.astype(np.uint8)
            # print(f'shadow consume time: {time.time() - start_shadow }')
        return img

    @property
    def targets_as_params(self):
        return ["image", "enlight"]

    def get_params_dependent_on_targets(self, params):
        enlight = params.get('enlight', False)
        return {'enlight': enlight}

    def get_transform_init_args_names(self):
        return ()


class Shadow_v2(Shadow):
    def __init__(self, strength_light=129, always_apply=False, radius_range=None, p=1., weight=0.5):
        super(Shadow_v2, self).__init__(strength_light=strength_light,
                                        always_apply=always_apply, radius_range=radius_range, p=p)
        self.weight = weight if weight < 0 else -weight

    def apply(self, img, **params):
        if not params.get('enlight', False):
            height, width, channel = img.shape

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the brightest pixel in the image
            center_loc = (np.random.randint(height), np.random.randint(width))

            min_radius = max(10, self.radius_range[0])
            max_radius = int(min(height, width)) if self.radius_range[1] == -1 else min(int(min(height, width)),
                                                                                        self.radius_range[1])
            radius = np.random.randint(min_radius, max_radius, 1)
            radius = int(radius[0])

            # Create a circle at the location of the brightest pixel
            circle_img = np.zeros_like(gray)
            cv2.circle(circle_img, center_loc, radius, (self.strength_light, self.strength_light, self.strength_light), -1)

            # Apply a Gaussian blur to the circle to create a light source effect
            blur_img = cv2.GaussianBlur(circle_img, (101, 101), 0)
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2RGB)
            # Add the blurred circle to the original image
            result = cv2.addWeighted(img, 1, blur_img, self.weight, 0)
            return result
        else:
            return img


class LightTransform(ImageOnlyTransform):
    def __init__(self, strength_light=150, lighter_prob=0.5, strength_dark=80, weight=0.5,
                 always_apply=True, radius_range=None, p=1.):
        if radius_range is None:
            radius_range = [10, -1]
        else:
            assert len(radius_range) == 2, 'radius range length should be 2'
            radius_range: list[int, int]
            if radius_range[1] != -1:
                assert radius_range[0] < radius_range[1], 'radius_range[0] should less than radius_range[1]'
        super().__init__(always_apply, p)
        self.strength_light = strength_light
        self.strength_dark = strength_dark
        self.lighter_prob = lighter_prob
        self.radius_range = radius_range
        self.weight = weight

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        do_lighter = True

        light_prob = np.random.random()
        if light_prob > self.lighter_prob:
            do_lighter = False

        height, width, channel = img.shape

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the brightest pixel in the image
        center_loc = (np.random.randint(height), np.random.randint(width))

        min_radius = max(10, self.radius_range[0])
        max_radius = int(min(height, width)) if self.radius_range[1] == -1 else min(int(min(height, width)),
                                                                                    self.radius_range[1])
        radius = np.random.randint(min_radius, max_radius, 1)
        radius = int(radius[0])

        # Create a circle at the location of the brightest pixel
        circle_img = np.zeros_like(gray)
        if do_lighter:
            strength = np.random.randint(self.strength_light)
        else:
            strength = np.random.randint(self.strength_dark)
        cv2.circle(circle_img, center_loc, radius, (strength, strength, strength), -1)

        # Apply a Gaussian blur to the circle to create a light source effect
        blur_img = cv2.GaussianBlur(circle_img, (101, 101), 0)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2RGB)

        # Add the blurred circle to the original image
        if do_lighter:
            result = cv2.addWeighted(img, 1, blur_img, self.weight, 0)
        else:
            result = cv2.addWeighted(img, 1, blur_img, -self.weight, 0)

        return result

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()

