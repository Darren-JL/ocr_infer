#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Time    : 2019/11/4 0004 21:38
# @Author  : peichao.xu
# @Email   : xj563853580@outlook.com
# @File    : ecs.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import os
import time
import pickle
import hashlib

PATH = os.path.realpath(__package__)
ECS_FILE = os.path.join(PATH, 'ecs.pb')
SECRET = "77812214"
MAX_DAYS = 50
TIME_FORMAT = "%Y-%m-%d"


def __encrypt_init(ptime=None):
    if ptime is None:
        ftime = time.strftime(TIME_FORMAT, time.localtime())
    else:
        pstruct = time.strptime(ptime, TIME_FORMAT)
        ftime = time.strftime(TIME_FORMAT, pstruct)
    data = {
        'secret': SECRET,
        'ftime': [ftime],
    }
    with open(ECS_FILE, 'wb') as fout:
        pickle.dump(data, fout)


def encrypt_check():
    ftime = time.strftime(TIME_FORMAT, time.localtime())

    try:
        with open(ECS_FILE, 'rb') as fin:
            data = pickle.load(fin)
        if data['secret'] != SECRET:
            raise RuntimeError("许可失效")
        if len(data['ftime']) > MAX_DAYS:
            raise RuntimeError("许可失效")
        if ftime not in data['ftime']:
            data['ftime'].append(ftime)
            with open(ECS_FILE, 'wb', encoding='utf-8') as fout:
                pickle.dump(data, fout)
        # print('剩余天数', MAX_DAYS - len(data['ftime']))
    except Exception:
        raise RuntimeError("许可失效")

# def __encrypt_sha512_init(ptime=None):
#     if ptime is None:
#         ftime = time.strftime(TIME_FORMAT, time.localtime())
#     else:
#         pstruct = time.strptime(ptime, TIME_FORMAT)
#         ftime = time.strftime(TIME_FORMAT, pstruct)
#     data = {
#         'secret': SECRET,
#         'ftime': [ftime],
#     }
#     data_bytes = str(data).encode(encoding='utf-8')
#
#     sha512 = hashlib.sha512(data_bytes).hexdigest()
#
#     sha512_bytes = sha512.encode(encoding='utf-8')
#     print(sha512)
#     with open(ECS_FILE, 'wb') as fout:
#         fout.write(sha512_bytes)
#
#
# def encrypt_sha1_check():
#     ftime = time.strftime(TIME_FORMAT, time.localtime())
#
#     try:
#         with open(ECS_FILE, 'rb') as fin:
#             sha1 = fin.read().decode('utf-8')
#
#         if data['secret'] != SECRET:
#             raise RuntimeError("许可失效")
#         if len(data['ftime']) > MAX_DAYS:
#             raise RuntimeError("许可失效")
#         if ftime not in data['ftime']:
#             data['ftime'].append(ftime)
#             with open(ECS_FILE, 'wb', encoding='utf-8') as fout:
#                 pickle.dump(data, fout)
#         print('剩余天数', MAX_DAYS - len(data['ftime']))
#     except Exception:
#         raise RuntimeError("许可失效")
