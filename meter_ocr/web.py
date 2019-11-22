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

# @Time    : 2019/11/14 0014 19:44
# @Author  : peichao.xu
# @Email   : xj563853580@outlook.com
# @File    : web.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import cv2
import time
import json
import base64

import meter_ocr
import socket

import tornado.web
import tornado.gen

import numpy as np

from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

from meter_ocr.ifs import Interface

IFS = Interface()

print('OCRService {} at {} (Press CTRL+C to quit) '.format(
    meter_ocr.__version__,
    str(socket.gethostname())))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class OCRService(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(50)

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    @run_on_executor
    def predict(self):
        info_dict = self.request.body_arguments
        has_det = self.get_argument('has_det', 'false')
        has_det = True if has_det.lower() == 'true' else False

        res = {
            'code': 0,
            'ftime': time.strftime("%Y-%m-%d %H:%M:%S %p %w", time.localtime()),
            'message': 'success',
            'result': {
                'time': 0,
                'bndbox': [],
                'polygon': [],
                'text': []
            },
        }
        try:
            im_base64 = info_dict['image'][0].split(b',')[-1]
            im_bytes = base64.b64decode(im_base64)
            np_array = np.frombuffer(im_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            pred = IFS._predict(image, has_det=has_det)
            res['code'] = pred['code']
            res['message'] = pred['message']
            res['result']['time'] = pred['time']
            res['result']['bndbox'] = pred['bndbox']
            res['result']['polygon'] = pred['polygon']
            res['result']['text'] = pred['text']
        except Exception as e:
            res['code'] = -1
            res['message'] = str(e)
        res = json.dumps(res, cls=MyEncoder)
        return res

    # @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        res = yield self.predict()
        self.write(res)
        self.finish()

    def get(self):
        return self.post()
