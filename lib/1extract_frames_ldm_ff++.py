#!/usr/bin/env python3
import torch
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
            .run(capture_stdout=True,quiet=True)
        )

        out = np.asarray(bytearray(out), dtype="uint8")
        out = cv2.imdecode(out, cv2.IMREAD_COLOR)
        return out

    def get_video_info(self, in_file):
        probe = ffmpeg.probe(in_file)
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream


VIDEO_PATH = "./data/FaceForensics++"
SAVE_IMGS_PATH = "./test_FF"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
DATASETS = {'Original', 'FaceSwap', 'FaceShifter',
            'Face2Face', 'Deepfakes', 'NeuralTextures'}
COMPRESSION = {'c23'}
NUM_FRAMES = 5
IMG_META_DICT = dict()


def parse_video_path(dataset, compression):
    # this path setting follows FF++ dataset
    if dataset == 'Original':
        dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
    elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
    else:
        raise NotImplementedError
    # get all videos under the specific manipulated/original sequences
    movies_path_list = sorted(glob(dataset_path+'*.mp4'))
    print("{} : videos are exist in {}".format(len(movies_path_list), dataset))
    return movies_path_list


def parse_labels(video_path):
    label = None
    if "original" in video_path:
        label = 0
    else:
        label = 1
    return label


def parse_source_save_path(save_path):
    source_save_path = None
    if "original" in save_path:
        source_save_path = save_path
    else:
        img_meta = save_path.split('/')
        source_target_index = img_meta[-1]
        source_index = source_target_index.split('_')[0]
        manipulation_name = img_meta[-4]
        original_name = "youtube"
        source_save_path = save_path.replace(
            "manipulated_sequences", "original_sequences"
        ).replace(
            manipulation_name, original_name
        ).replace(
            source_target_index, source_index
        )
    return source_save_path


def preprocess_video(video_path, save_path, face_detector, face_predictor):
    # save the video meta info here
    video_dict = dict()
    # get the labels
    label = parse_labels(video_path)
    # get the path of corresponding source imgs
    source_save_path = parse_source_save_path(save_path)
    # prepare the save path
    os.makedirs(save_path, exist_ok=True)
    # read the video and prepare the sampled index

    videoset = VideoSet(video_path)
    videoLoader = DataLoader(videoset)
    # process each frame
    for cnt_frame, frame in enumerate(videoLoader):
        frame=frame[0].numpy()
        # print(type(frame))
        # print(frame.shape)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2.imwrite('./my_test1.jpg', frame)
        faces = face_detector.predict_jsons(frame)
        # print(faces)
        if len(faces) == 0:

            continue
        # faces=faces[0]

        # cv2.rectangle(frame, (int(faces[0]), int(faces[1])), (int(faces[2]), int(faces[3])), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.imwrite('./my_test.jpg',frame)
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        land_fives = list()
        for face_idx in range(len(faces)):
            score = faces[face_idx]['score']
            if score <= 0.8:
                # print('skip this image ')
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
        video_dict['label'] = label
        video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"

        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict
        # save one frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
    with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)
    return


def main():
    face_detector = get_model("resnet50_2020-07-20",
                              max_size=2048, device=torch.device('cuda:0'))
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    for dataset in DATASETS:
        for comp in COMPRESSION:
            movies_path_list = parse_video_path(dataset, comp)
            n_sample = len(movies_path_list)
            for i in tqdm(range(n_sample)):
                save_path_per_video = movies_path_list[i].replace(
                    VIDEO_PATH, SAVE_IMGS_PATH
                ).replace('.mp4', '').replace("/videos", "/frames")
                preprocess_video(
                    movies_path_list[i], save_path_per_video,
                    face_detector, face_predictor
                )
    with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
