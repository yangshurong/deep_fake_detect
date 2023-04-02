#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input


class DeepfakeDatasetFF(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        assert mode in ['train', 'test']
        self.root = self.config[mode]['img_path']
        self.landmark_path = self.config[mode]['ld_path']
        self.rng = np.random
        self.do_train = True if mode == 'train' else False
        self.samples = self.collect_samples()

    def collect_samples(self) -> List:
        samples = []
        # directory = os.path.expanduser(self.root)
        from_res = {}
        from_res = {}
        with open(self.landmark_path, 'r', encoding='utf-8') as f:
            s = f.read()
            from_res = json.loads(s)

        for k, v in from_res.items():
            video_path = k
            source_path = v['source_path']
            if v['img_path'] is not None:
                video_path = os.path.join(v['img_path'], k)
                source_path = os.path.join(v['img_path'], v['source_path'])
            if not os.path.exists(video_path) or not os.path.exists(source_path):
                continue
            samples.append(
                (video_path, {'labels': int(v['label']), 'landmark': v['landmark'],
                              'source_path': source_path,
                              'video_name': k.split('/')[0]})
            )
        print(f'get {len(samples)} train images')
        return samples

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if self.mode == "train":
            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = torch.Tensor(img.transpose(2, 0, 1))
            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']
            return img, (label, video_name)

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


# if __name__ == "__main__":
#     from lib.util import load_config
#     config = load_config('./configs/caddm_train.cfg')
#     d = DeepfakeDataset(mode="test", config=config)
#     for index in range(len(d)):
#         res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
