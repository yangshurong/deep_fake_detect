#!/usr/bin/env python3
import time
import ffmpeg
from multiprocessing.pool import ThreadPool
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
# torch.multiprocessing.set_start_method('spawn')
from trainlog import get_logger
LOGGER = get_logger()


class VideoSet(Dataset):
    def __init__(self, cap_video):
        self.cap_video = cap_video
        # source_video = cv2.VideoCapture(source_path)
        self.frame_count_video = int(
            self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.frame_count_video

    def __getitem__(self, idx):
        ret, frame = self.cap_video.read()

        return ret, frame, idx


SAVE_IMGS_PATH = "./test_FF"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
VIDEO_PATH_LIST = ['./data/FaceForensics++/manipulated_sequences/Deepfakes/c23/videos',
                   './data/FaceForensics++/manipulated_sequences/Face2Face/c23/videos',
                   './data/FaceForensics++/manipulated_sequences/FaceSwap/c23/videos',
                   './data/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos',
                   './data/FaceForensics++/original_sequences/youtube/c23/videos']
NUM_FRAMES = 5
NUM_PROCESSES = 8
SAVE_INTERVAL = 5


def preprocess_video(video_name, source_name, video_path, source_path, label, face_detector, face_predictor, cur_num):
    # save the video meta info here

    # video_name only name
    # source_name only name
    # video_path source_path video path of source and video
    # print(f'{video_name}/frame_1.png')

    # get the path of corresponding source imgs
    save_path = os.path.join(SAVE_IMGS_PATH, video_name)
    if os.path.exists(os.path.join(save_path, 'ldm.json')):
        return
    os.makedirs(save_path, exist_ok=True)

    # videoset = VideoSet(video_path)
    # videoLoader = DataLoader(videoset, num_workers=0)
    try:

        probe = ffmpeg.probe(video_path)
        video_info = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        total_frames = int(video_info['nb_frames'])
        frame_idxs = np.linspace(0, total_frames - 1,
                                 NUM_FRAMES, endpoint=True, dtype=np.int32)
    except:
        return
    LOGGER.info(f'start work for {cur_num}')

    tmp_res = {}
    for cnt_frame, idxs in enumerate(frame_idxs):
        out, err = (
            ffmpeg.input(video_path)
            .filter('select', 'gte(n,{})'.format(idxs))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, quiet=True)
        )

        out = np.asarray(bytearray(out), dtype="uint8")
        out = cv2.imdecode(out, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

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
        video_dict = {}
        video_dict['dets'] = land_fives.tolist()
        video_dict['landmark'] = landmarks.tolist()
        video_dict['label'] = label
        video_dict['source_path'] = f"{source_name}/frame_{cnt_frame}.png"
        tmp_res[f"{video_name}/frame_{cnt_frame}.png"] = video_dict
        # print(f"{video_name}/frame_{cnt_frame}")
        # print(video_dict)
        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)

    with open(f"{save_path}/ldm.json", 'w') as f:
        json.dump(tmp_res, f)
        LOGGER.info(f"finish work for {cur_num}")
    return
    # return tmp_res, cur_num, save_path


def run_page(face_detector, face_predictor):
    cur_num = 0
    my_pool = ThreadPool(processes=NUM_PROCESSES)
    for video_path in VIDEO_PATH_LIST:
        label = 1
        pre_name = video_path.split('/')[-3]
        if video_path.find('manipulated') != -1:
            label = 0
        g = os.walk(video_path)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                # print(os.path.join(path, file_name))
                source_path = file_name

                cur_path = os.path.join(path, file_name)
                if label == 0:
                    video_name = pre_name+file_name[4:7]
                    origin = 'youtube'+file_name[0:3]
                    source_path = os.path.join(VIDEO_PATH_LIST[4], origin+'.mp4')
                else:
                    video_name = pre_name+file_name[0:3]
                    origin = 'youtube'+file_name[0:3]
                    source_path = os.path.join(VIDEO_PATH_LIST[4], origin+'.mp4')
                if os.path.exists(cur_path) == False:
                    # LOGGER.info(cur_path)
                    continue
                my_pool.apply_async(func=preprocess_video, args=(video_name, origin.split('.')[0],
                                                                 cur_path, source_path, label,
                                                                 face_detector, face_predictor, cur_num))
                # preprocess_video(video_name, origin,
                #                  cur_path, source_path, label,
                #                  face_detector, face_predictor, cur_num
                #                  )                                                 
                if cur_num % (NUM_PROCESSES+2) == 0:
                    my_pool.close()
                    my_pool.join()
                    my_pool = ThreadPool(processes=NUM_PROCESSES)
                cur_num += 1


def main():
    face_detector = get_model("resnet50_2020-07-20",
                              max_size=2048, device=torch.device('cuda:0'))
    face_detector.eval()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    run_page(face_detector, face_predictor)


if __name__ == '__main__':
    res_dict = dict()
    main()
# vim: ts=4 sw=4 sts=4 expandtab
