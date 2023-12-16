import numbers
import os
import queue as Queue
import threading
import time
from typing import Iterable
import cv2
import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2
from insightface.app import MaskAugmentation
from data_aug import *


def get_dataloader(
        root_dir,
        local_rank,
        batch_size,
        dali=False,
        dali_aug=False,
        seed=2048,
        num_workers=2,
        aug_pipeline=None,
        add_rec=None,
        keep_max_yaw_val=None,
        keep_max_pitch_val=None,
        keep_max_yaw_and_pitch_val=None,
) -> Iterable:
    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank,
                                  aug_pipeline=aug_pipeline,
                                  add_rec=add_rec,
                                  keep_max_yaw_val=keep_max_yaw_val,
                                  keep_max_pitch_val=keep_max_pitch_val,
                                  keep_max_yaw_and_pitch_val=keep_max_yaw_and_pitch_val,
                                  )

    # Image Folder
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, aug_pipeline=None, add_rec=None, keep_max_yaw_val=None,
                 keep_max_pitch_val=None, keep_max_yaw_and_pitch_val=None):
        super(MXFaceDataset, self).__init__()

        if keep_max_yaw_val is not None:
            assert isinstance(keep_max_yaw_val, (int, float)), 'add_yaw should be a int or a float'
        if keep_max_pitch_val is not None:
            assert isinstance(keep_max_pitch_val, (int, float)), 'add_pitch should be a int or a float'
        if keep_max_yaw_and_pitch_val is not None:
            assert isinstance(keep_max_yaw_and_pitch_val, (list, tuple)), 'add_yaw_pitch should be a list or a tuple'

        if keep_max_yaw_and_pitch_val is not None:
            if len(keep_max_yaw_and_pitch_val) == 1:
                real_keep_max_yaw_and_pitch_val = [keep_max_yaw_and_pitch_val[0], keep_max_yaw_and_pitch_val[0]]
            else:
                real_keep_max_yaw_and_pitch_val = keep_max_yaw_and_pitch_val

        # read data augment method
        aug_cls = {'MaskAugmentationAdd': MaskAugmentationAdd, 'CutOut': CutOut, 'GridMask': GridMask,
                   'BlurByBlock': BlurByBlock, 'Enlight': Enlight, 'Shadow': Shadow, 'LightTransform': LightTransform}
        transform_list = []
        assert aug_pipeline is not None, f'aug_pipeline: {aug_pipeline} is none'
        for aug_name, prop in aug_pipeline.items():
            assert aug_name in aug_cls, f' not support {aug_name} in data aug'
            transform_list.append(aug_cls[aug_name](**prop))

        if local_rank == 0:
            print('data_transform_list:', transform_list)

        transform_list += \
            [
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        # here, the input for A transform is rgb cv2 img
        self.transform = A.Compose(
            transform_list
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

        # add warp image rec path
        self.imgidx_add = []
        if add_rec is not None:
            path_imgrec_add = os.path.join(add_rec, 'glint_part1.rec')
            path_imgidx_add = os.path.join(add_rec, 'glint_part1.idx')
            assert os.path.exists(path_imgidx_add), f'{path_imgidx_add} not exist'
            assert os.path.exists(path_imgrec_add), f'{path_imgrec_add} not exist'
            self.imgrec_add = mx.recordio.MXIndexedRecordIO(path_imgidx_add, path_imgrec_add, 'r')
            imgidx_add_tmp = np.array(self.imgrec_add.keys)
            start_time = time.time()
            for idx in imgidx_add_tmp:
                s = self.imgrec_add.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                hlabel = header.label

                warp_pitch, warp_yaw, warp_roll = hlabel[-3:] * 180 / np.pi
                flag = (keep_max_yaw_val and warp_yaw > keep_max_yaw_val and warp_pitch == 0) or \
                       (keep_max_pitch_val and warp_yaw == 0 and warp_pitch > keep_max_pitch_val) or \
                       (keep_max_yaw_and_pitch_val and warp_yaw > real_keep_max_yaw_and_pitch_val[0] and
                        warp_pitch > real_keep_max_yaw_and_pitch_val[1])
                if not flag:
                    self.imgidx_add.append(idx)
            print(f'for loop strip consume time: {time.time() - start_time}')

        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        # print(header)
        # print(len(self.imgrec.keys))
        if header.flag > 0:
            if len(header.label) == 2:
                self.imgidx = np.array(range(1, int(header.label[0])))
            else:
                self.imgidx = np.array(list(self.imgrec.keys))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        # print('imgidx len:', len(self.imgidx))

    def __getitem__(self, index):
        # get img by idx
        if index < len(self.imgidx):
            real_idx = self.imgidx[index]
            s = self.imgrec.read_idx(real_idx)
        else:
            index = index - len(self.imgidx)
            real_idx = self.imgidx_add[index]
            s = self.imgrec_add.read_idx(real_idx)

        header, img = mx.recordio.unpack(s)
        hlabel = header.label
        # print('hlabel:', hlabel.__class__)
        sample = mx.image.imdecode(img).asnumpy()
        if not isinstance(hlabel, numbers.Number):
            idlabel = hlabel[0]
        else:
            idlabel = hlabel
        label = torch.tensor(idlabel, dtype=torch.long)
        if self.transform is not None:
            sample = self.transform(image=sample, hlabel=hlabel)['image']

        # import cv2
        # cv2.imwrite("./datasets.png", sample[:,:,::-1])

        return sample, label

    def __len__(self):
        return len(self.imgidx) + len(self.imgidx_add)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
        batch_size: int, rec_file: str, idx_file: str, num_threads: int,
        initial_fill=32768, random_shuffle=True,
        prefetch_queue_depth=1, local_rank=0, name="reader",
        mean=(127.5, 127.5, 127.5),
        std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill,
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
