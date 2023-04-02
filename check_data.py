# -*- coding: utf-8 -*-
import lmdb
import pickle
env_db = lmdb.open('./test_DFDC_lmdb')
# env_db = lmdb.open("./trainC")

txn = env_db.begin()
img = txn.get('./test_DFDC/aszdwxqtnb/frame_0.png'.encode('ascii'))
# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None
img = pickle.loads(img)

env_db.close()
