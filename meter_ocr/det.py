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

# @Time    : 2019/11/2 0002 13:52
# @Author  : peichao.xu
# @Email   : xj563853580@outlook.com
# @File    : detector.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import tensorflow as tf


class DET(object):
    def __init__(self, weights, allow_growth=True, allow_soft_placement=True, log_device_placement=False):
        session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                        log_device_placement=log_device_placement)
        session_config.gpu_options.allow_growth = allow_growth

        self.threshold = 0.5

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=session_config)

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with open(weights, "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

    def _resize(self, img, size=(300, 300)):
        img_h, img_w, _ = img.shape
        pad = [0, 0]
        if img_h > img_w:
            scale = img_h * 1.0 / size[0]
            img_h = size[1]
            img_w = int(img_w * 1.0 / scale)
            pad[1] = (size[0] - img_w) // 2
        else:
            scale = img_w * 1.0 / size[1]
            img_w = size[0]
            img_h = int(img_h * 1.0 / scale)
            pad[0] = (size[1] - img_h) // 2
        tmp_image = cv2.resize(img, (img_w, img_h))
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
        bk_image = np.zeros([size[1], size[0], 3], dtype=np.uint8)
        bk_image[pad[0]:pad[0] + img_h, pad[1]:pad[1] + img_w] = tmp_image
        return bk_image, pad, scale

    def set_threshold(self, value: float):
        self.threshold = value

    def visual(self, image, boxes, save=False, show=False):
        import matplotlib.pyplot as plt
        for box in boxes:
            x0, y0, x1, y1, cls, score = box
            cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        plt.imshow(image)
        if save:
            plt.imsave('det_demo.jpg')
        if show:
            plt.show()

    def predict(self, img):
        '''
        :param img: bgr 3 channel cv2 mat
        :return: [x0, y0, x1, y1, cls_idx, score]
        '''
        image, pad, scale = self._resize(img)
        result = self.session.run(["outputs:0"], feed_dict={"inputs_image:0": np.float32([image])})[0]
        keep_boxes = []
        for det in result:
            box = det[:4]
            box -= [pad[1], pad[0], pad[1], pad[0]]
            box *= scale
            x0, y0, x1, y1 = box.astype(np.int)
            cls_idx = int(det[4])
            score = det[5]
            if score < self.threshold:
                continue
            keep_boxes.append([x0, y0, x1, y1, cls_idx, score])
        return keep_boxes


if __name__ == '__main__':
    det_model = DET('/code/disk1/xupeichao/data/dianbiao/data_v1/ssd/v2/model.pb')

    image = cv2.imread('/code/disk1/xupeichao/data/dianbiao/data_v1/done/img/65d107c8-f8ba-4595-bb0b-d8d9b64418f7.jpg')

    boxes = det_model.predict(image)

    det_model.visual(image, boxes)
