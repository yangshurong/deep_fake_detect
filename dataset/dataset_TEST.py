#!/usr/bin/env python3
from PIL import Image
import random
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import albumentations as alb
from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input


class DeepfakeDatasetTEST(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, set_name, root, cfg):
        super().__init__()
        self.root = root
        self.config = cfg
        self.set_name = set_name
        self.rng = np.random
        self.samples = self.collect_samples()

    def collect_samples(self) -> List:
        samples = []
        pre_len = 0
        # directory = os.path.expanduser(self.root)
        pre_len = len(samples)
        true_num = 0
        for path, dir_list, file_list in os.walk(self.root):
            for dir_name in dir_list:
                # if not os.path.exists(os.path.join(path, dir_name, 'existF.json')):
                #     continue
                ldm_path = os.path.join(path, dir_name, 'ldm.json')
                if os.path.exists(ldm_path) == False:
                    continue

                from_res = {}
                with open(ldm_path, 'r', encoding='utf-8') as f:
                    s = f.read()
                    from_res = json.loads(s)

                for k, v in from_res.items():
                    video_path = os.path.join(path, k)
                    source_path = os.path.join(path, v['source_path'])
                    if not os.path.exists(video_path):
                        continue

                    if not os.path.exists(source_path):
                        if int(v['label']) == 0:
                            continue
                    # if int(v['label']) == 0 and set_name != 'FF++' and np.random.uniform(0, 1) <= 0.8:
                    #     continue
                    samples.append(
                        (video_path, {'labels': int(v['label']) ^ 1, 'landmark': v['landmark'],
                                      'source_path': source_path,
                                      'video_name': k.split('/')[0],
                                      'use_SBI': True,
                                      'dets': v['dets']})
                    )

                    if v['label'] == 1:
                        true_num += 1

            break
        print(
            f'get {self.set_name} {len(samples)-pre_len} images for test')
        print(
            f'get {self.set_name} {true_num} true images for test')
        return samples

    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def get_cache(self, path):
        if path in self.images_cache:
            return self.images_cache[path]
        if len(self.images_cache) < self.config['cache_num']:
            self.images_cache[path] = np.array(Image.open(path))
            return self.images_cache[path]
        return np.array(Image.open(path))

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        img, label_dict = prepare_test_input(
            [img], ld, label, self.config
        )
        
        img = torch.Tensor(img[0].transpose(2, 0, 1))
        video_name = label_meta['video_name']
        return img, (label, video_name)

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        img_f, img_r = zip(*batch)
        data = {}
        data['img'] = torch.cat(
            [torch.tensor(img_r).float(), torch.tensor(img_f).float()], 0)
        data['label'] = torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


# if __name__ == "__main__":
#     from lib.util import load_config
#     config = load_config('./configs/caddm_train.cfg')
#     d = DeepfakeDataset(mode="test", config=config)
#     for index in range(len(d)):
#         res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
