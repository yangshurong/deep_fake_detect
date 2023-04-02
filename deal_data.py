import pickle
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import os
import json
import lmdb
import numpy as np
OBJECT_PATH = './test_DFDC'
LMDB_PATH = './test_dfdc_lmdb'
os.makedirs(LMDB_PATH, exist_ok=True)
REMOVE = False
db = lmdb.open(LMDB_PATH, subdir=True, readonly=False,
               map_size=1099511627776 * 2,
               meminit=False, map_async=True)
txn = db.begin(write=True)
SAMPLES = {}
for path, dir_list, file_list in os.walk(OBJECT_PATH):
    for dir_name in dir_list:
        # if os.path.exists(os.path.join(path, dir_name, 'exist.json')):
        #     continue
        ldm_path = os.path.join(path, dir_name, 'ldm.json')
        if os.path.exists(ldm_path) == False:
            continue
        from_res = {}
        with open(ldm_path, 'r', encoding='utf-8') as f:
            s = f.read()
            from_res = json.loads(s)

        label_type = None
        for k, v in from_res.items():
            label_type = v['label']

        if REMOVE and np.random.rand() < 0.9 and label_type == 0:
            
            continue

        for k, v in from_res.items():
            video_path = os.path.join(path, k)
            source_path = os.path.join(path, v['source_path'])

            if not os.path.exists(video_path):
                continue

            if not os.path.exists(source_path):
                if int(v['label']) == 0:
                    continue

            if video_path not in SAMPLES:
                SAMPLES[video_path] = 1
            if source_path not in SAMPLES:
                SAMPLES[source_path] = 1

        # with open(os.path.join(path, dir_name, 'exist.json'), 'w', encoding='utf-8') as f:
        #     f.write(json.dumps({'a': 'b'}))

    break


class utilsDataset(Dataset):

    def __init__(self, samples):
        super().__init__()

        self.samples = []
        for k, v in samples.items():
            self.samples.append(k)

    def __getitem__(self, index):
        path = self.samples[index]
        # print(path)
        img = np.array(Image.open(path))
        return path, img

    def __len__(self):
        return len(self.samples)


# print(len(SAMPLES))
train_dataset = utilsDataset(SAMPLES)
train_loader = DataLoader(train_dataset,
                          batch_size=1, num_workers=8
                          )


for i, (path, img) in enumerate(train_loader):
    print(f'---------{path}------------')
    txn.put(path[0].encode('ascii'), pickle.dumps(img[0].numpy()))
    # txn.put(path[0], img[0].numpy())
    txn.commit()
    txn = db.begin(write=True)
db.sync()
db.close()
