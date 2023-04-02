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


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='train'):
    assert phase in ['train', 'val', 'test']

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1-x0
        h = y1-y0
        w0_margin = w/4  # 0#np.random.rand()*(w/8)
        w1_margin = w/4
        h0_margin = h/4  # 0#np.random.rand()*(h/5)
        h1_margin = h/4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1-x0
        h = y1-y0
        w0_margin = w/8  # 0#np.random.rand()*(w/8)
        w1_margin = w/8
        h0_margin = h/2  # 0#np.random.rand()*(h/5)
        h1_margin = h/5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        w1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0-h0_margin))
    y1_new = min(H, int(y1+h1_margin)+1)
    x0_new = max(0, int(x0-w0_margin))
    x1_new = min(W, int(x1+w1_margin)+1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p-x0_new, q-y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p-x0_new, q-y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W/r), int(H/r)),
                            interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds


def alpha_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def dynamic_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(
        mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
    kernel_list = [(11, 11), (9, 9), (7, 7), (5, 5), (3, 3)]
    blend_list = [0.25, 0.5, 0.75]
    kernel_idxs = random.choices(range(len(kernel_list)), k=2)
    blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
    mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
    # print(mask_blured.max())
    mask_blured[mask_blured < mask_blured.max()] = 0
    mask_blured[mask_blured > 0] = 1
    # mask_blured = mask
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
    mask_blured = mask_blured/(mask_blured.max())
    return mask_blured.reshape((mask_blured.shape+(1,)))


class DeepfakeDatasetSBI(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        assert mode in ['train', 'test']
        self.config = config
        self.mode = mode
        self.use_sbi = config[mode]['use_SBI']
        self.rng = np.random
        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.samples = self.collect_samples()
        self.do_train = True if mode == 'train' else False
        self.images_cache = {}

    def collect_samples(self) -> List:
        samples = []
        pre_len = 0
        # directory = os.path.expanduser(self.root)
        for set_name, set_value in self.config[self.mode]['dataset'].items():
            pre_len = len(samples)
            true_num = 0
            for path, dir_list, file_list in os.walk(set_value):
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
                        if int(v['label']) == 1 and self.mode == 'train':
                            samples.append(
                                (video_path, {'labels': int(v['label']) ^ 1, 'landmark': v['landmark'],
                                              'source_path': source_path,
                                              'video_name': k.split('/')[0],
                                              'use_SBI': False,
                                              'dets': v['dets']})
                            )

                        if v['label'] == 1:
                            true_num += 1

                break
            print(
                f'get {set_name} {len(samples)-pre_len} images for {self.mode}')
            print(
                f'get {set_name} {true_num} true images for {self.mode}')
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
        img = None
        source_img = None
        landmark_cropped = None

        if self.use_sbi and label == 0 and self.mode == 'train':
            img = np.array(Image.open(path))
            # img = self.get_cache(path)
            bboxes = np.array(label_meta['dets'])
            bbox_lm = np.array([ld[:, 0].min(), ld[:, 1].min(),
                                ld[:, 0].max(), ld[:, 1].max()])
            iou_max = -1
            for i in range(len(bboxes)):
                iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                if iou_max < iou:
                    bbox = bboxes[i]
                    iou_max = iou

            landmark = self.reorder_landmark(ld)
            # if self.mode == 'train':
            # if np.random.rand() < 0.5:
            # img, _, landmark, bbox = self.hflip(img, None, landmark, bboxes)

            img, landmark, bbox, __ = crop_face(
                img, landmark, bbox, margin=True, crop_by_bbox=False)

            img_r, img_f, mask_f = self.self_blending(
                img.copy(), landmark.copy())

            if self.mode == 'train':
                transformed = self.transforms(image=img_f.astype(
                    'uint8'), image1=img_r.astype('uint8'))
                img_f = transformed['image']
                img_r = transformed['image1']

            img, landmark_cropped, bbox_cropped, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                img_f, landmark, bbox, margin=False, crop_by_bbox=True, abs_coord=True, phase=self.mode)

            source_img = img_r[y0_new:y1_new, x0_new:x1_new]
        else:
            # img = self.get_cache(path)
            # source_img = self.get_cache(source_path)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        # img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
        # img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255

        # img=img_f.transpose((2,0,1))
        # source_img=img_r.transpose((2,0,1))
        # flag=False

        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        # source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if self.mode == "train":
            if landmark_cropped is not None:
                img, label_dict = prepare_train_input(
                    img, source_img, landmark_cropped, label, self.config, self.do_train
                )
            else:
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
            if landmark_cropped is not None:
                img, label_dict = prepare_test_input(
                    [img], landmark_cropped, label, self.config
                )
            else:
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

    def get_fuck(self):
        return 'i have no idea'

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                        alb.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=1),
                        alb.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
                        ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(
                hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1/0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]

        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(
                image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = dynamic_blend(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W-landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W-bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W-bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W-bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W-bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W-bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W-bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

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
