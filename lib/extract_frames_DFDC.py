#!/usr/bin/env python3
import torch
from retinaface.pre_trained_models import get_model
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imutils import face_utils

import ffmpeg


class VideoSet(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        video_info = self.get_video_info(video_path)
        self.total_frames = int(video_info['nb_frames'])
        self.frame_idxs = np.linspace(0, self.total_frames - 1,
                                      NUM_FRAMES, endpoint=True, dtype=np.int32)

    def __len__(self):
        return len(self.frame_idxs)

    def __getitem__(self, idx):
        out, err = (
            ffmpeg.input(self.video_path)
            .filter('select', 'gte(n,{})'.format(self.frame_idxs[idx]))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, quiet=True)
        )

        out = np.asarray(bytearray(out), dtype="uint8")
        out = cv2.imdecode(out, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out

    def get_video_info(self, in_file):
        probe = ffmpeg.probe(in_file)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream


PAGE_ID_LIST = [1, 3]
SAVE_IMGS_PATH = "./test_DFDC"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
DATASETS = {'Original', 'FaceSwap', 'FaceShifter',
            'Face2Face', 'Deepfakes', 'NeuralTextures'}
COMPRESSION = {'c23'}
NUM_FRAMES = 5


def preprocess_video(video_name, source_name, video_path, source_path, label, face_detector, face_predictor, IMG_META_DICT):
    # save the video meta info here
    # video_name only name
    # source_name only name
    # video_path source_path video path of source and video
    # print(f'{video_name}/frame_1.png')
    if f'{video_name}/frame_0.png' in IMG_META_DICT:
        return
    if os.path.exists(video_path) == False:
        return
    video_dict = dict()
    # get the path of corresponding source imgs
    save_path = os.path.join(os.getcwd(), 'test_DFDC')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, video_name)
    os.makedirs(save_path, exist_ok=True)

    videoset = VideoSet(video_path)
    videoLoader = DataLoader(videoset)

    for cnt_frame, frame in enumerate(videoLoader):

        frame = frame[0].numpy()

        faces = face_detector.predict_jsons(frame)

        if len(faces) == 0:
            continue
        # faces=faces[0]
        # print(faces)
        # cv2.rectangle(frame, (int(faces[0]), int(faces[1])), (int(faces[2]), int(faces[3])), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.imwrite('./my_test.jpg',frame)
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        land_fives = list()
        for face_idx in range(len(faces)):
            score = faces[face_idx]['score']
            if score <= 0.8:
                continue
            x0, y0, x1, y1 = faces[face_idx]['bbox']
            land_five = np.array([[x0, y0], [x1, y1]] +
                                 faces[face_idx]['landmarks'])
            land_fives.append(land_five)

            dets = dlib.rectangle(int(x0), int(y0), int(x1), int(y1))
            landmark = face_predictor(frame, dets)
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_size = (x1 - x0) * (y1 - y0)
            size_list.append(face_size)
            landmarks.append(landmark)

        if len(landmarks) == 0:
            continue
        landmarks = np.concatenate(landmarks).reshape(
            (len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

        land_fives = np.concatenate(land_fives).reshape(
            (len(size_list),)+land_five.shape)
        land_fives = land_fives[np.argsort(np.array(size_list))[::-1]]
        # save the meta info of the video
        video_dict['dets'] = land_fives.tolist()
        video_dict['landmark'] = landmarks.tolist()
        video_dict['label'] = 1 if label != 'FAKE' else 0
        video_dict['source_path'] = f"{source_name}/frame_{cnt_frame}.png"
        IMG_META_DICT[f"{video_name}/frame_{cnt_frame}.png"] = video_dict
        # print(f"{video_name}/frame_{cnt_frame}")
        # print(video_dict)
        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)

    return


def run_page(page_id, face_detector, face_predictor, IMG_META_DICT):
    meta_path = './data/dfdc_train_part_{}/metadata.json'.format(str(page_id))
    from_res = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        s = f.read()
        from_res = json.loads(s)
    # sort for REAl at front
    res = {}
    for k, v in from_res.items():
        if v['label'] == 'REAL':
            res[k] = v
    for k, v in from_res.items():
        if v['label'] == 'FAKE':
            res[k] = v

    pre_video_path = os.path.join(
        os.getcwd(), './data/dfdc_train_part_{}/'.format(str(page_id)))
    cur_num = 0
    for k, v in tqdm(from_res.items()):
        label = v['label']
        split = v['split']
        origin = k
        if label == 'FAKE':
            origin = v['original']
        cur_path = os.path.join(pre_video_path, k)
        source_path = os.path.join(pre_video_path, origin)
        preprocess_video(k.split('.')[0], origin.split('.')[0],
                         cur_path, source_path, label,
                         face_detector, face_predictor, IMG_META_DICT
                         )
        if cur_num % 1 == 0:
            with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
                json.dump(IMG_META_DICT, f)
        cur_num += 1


def main():
    face_detector = get_model("resnet50_2020-07-20",
                              max_size=2048, device=torch.device('cuda:0'))
    face_detector.eval()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    IMG_META_DICT = dict()
    if os.path.exists('./test_DFDC/ldm.json'):
        with open('./test_DFDC/ldm.json', 'r', encoding='utf-8') as f:
            s = f.read()
            IMG_META_DICT = json.loads(s)
            print('finish read json', './test_DFDC/ldm.json')
    for i in PAGE_ID_LIST:

        run_page(i, face_detector, face_predictor, IMG_META_DICT)


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
