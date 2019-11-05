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

# @Time    : 2019/11/3 0003 16:14
# @Author  : peichao.xu
# @Email   : xj563853580@outlook.com
# @File    : detector.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np

from meter_ocr.det import DET
from meter_ocr.tps import TPS
from meter_ocr.rnn import RNN
from meter_ocr.ecs import encrypt_check

PATH = os.path.realpath(__package__)


class Interface(object):
    def __init__(self,
                 det_weights=os.path.join(PATH, 'det.pb'),
                 tps_weights=os.path.join(PATH, 'tps.pb'),
                 rnn_weights=os.path.join(PATH, 'rnn.pb'),
                 allow_growth=True,
                 allow_soft_placement=True,
                 log_device_placement=False):

        assert os.path.exists(det_weights), det_weights
        assert os.path.exists(tps_weights), tps_weights
        assert os.path.exists(rnn_weights), rnn_weights

        encrypt_check()

        self.det_model = DET(det_weights, allow_growth, allow_soft_placement, log_device_placement)
        self.tps_model = TPS(tps_weights, allow_growth, allow_soft_placement, log_device_placement)
        self.rnn_model = RNN(rnn_weights, allow_growth, allow_soft_placement, log_device_placement)

        self.message = {
            -1: 'unknown error',
            0: 'success',
            1: 'no target detected',
        }

    def crop_with_box(self, img, box, scale=1.25):
        assert isinstance(scale, float)
        src_h, src_w = img.shape[:2]
        x0, y0, x1, y1 = np.int32(box)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        nw = abs(x0 - x1) * scale
        nh = abs(y0 - y1) * scale
        xx0 = np.clip(int(cx - (nw * 0.5)), 0, src_w)
        yy0 = np.clip(int(cy - (nh * 0.5)), 0, src_h)
        xx1 = np.clip(int(xx0 + nw), 0, src_w)
        yy1 = np.clip(int(yy0 + nh), 0, src_h)
        roi = img[yy0:yy1, xx0:xx1, :].copy()
        return roi

    def _predict(self, img):
        '''param img: bgr, 3 channel, cv2.Mat'''
        result = {
            'code': 0,
            'time': 0,
            'message': self.message[0],
            'bndbox': [],
            'points': [],
            'text': [],
        }
        try:
            t_start = time.time()
            boxes = self.det_model.predict(img)
            if len(boxes) == 0:
                result['code'] = 1
                result['message'] = self.message[1]
                return result
            x0, y0, x1, y1, cls, score = boxes[0]
            result['bndbox'] = [x0, y0, x1, y1, score]
            roi = self.crop_with_box(img, [x0, y0, x1, y1], scale=1.25)
            points = self.tps_model.predict(roi)
            _pts = points.copy()
            _pts[:, 0] = _pts[:, 0] + x0
            _pts[:, 1] = _pts[:, 1] + y1
            result['points'] = _pts.tolist()
            tfm_img = self.tps_model.transform(roi, points, dst_size=(320, 32))
            text = self.rnn_model.predict(tfm_img)
            result['time'] = time.time() - t_start
            result['text'] = text[0]
        except Exception as e:
            print(e)
            result['code'] = -1
            result['message'] = self.message[-1]

        return result
