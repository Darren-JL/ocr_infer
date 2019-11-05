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
# @File    : pts.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import tensorflow as tf


class TPS(object):
    def __init__(self, weights, allow_growth=True, allow_soft_placement=True, log_device_placement=False):
        session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                        log_device_placement=log_device_placement)
        session_config.gpu_options.allow_growth = allow_growth

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=session_config)

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with open(weights, "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

    def _preprocess(self, img):
        image = cv2.resize(img, (224, 224))
        image = image.astype('float32')
        image /= 127.5
        image -= 1.
        return image

    def visual(self, img, points, save=False, show=False):
        import matplotlib.pyplot as plt
        if not points is None:
            cv2.polylines(img, np.int32([points]), 1, (0, 255, 0), 2)
        plt.imshow(img)
        if save: plt.imsave('det_demo.jpg')
        if show:
            plt.show()

    def transform(self, img, points, dst_size=(448, 32)):
        dst_points = np.array([[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]],
                              dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, dst_points)
        dst_img = cv2.warpPerspective(img, M, dst_size)
        return dst_img

    def predict(self, img):
        '''
        :param img: 3 channel bgr cv2 mat class
        :return: [lt, rt, rb, lb] shape is [4, 2]
        '''
        src_h, src_w = img.shape[:2]
        image = self._preprocess(img)
        points = self.session.run(["outputs:0"], feed_dict={"inputs:0": [image]})[0]
        points = np.squeeze(points).reshape((4, 2))
        points[:, 0] = src_w * (points[:, 0] + 0.5)
        points[:, 1] = src_h * (points[:, 1] + 0.5)
        return points


if __name__ == '__main__':
    import os

    tps_model = TPS('/code/disk1/xupeichao/data/dianbiao/data_v1/four_pts/v2/exports/model_2300.pb')

    image_dir = '/code/disk1/xupeichao/data/dianbiao/data_v1/four_points/images/'
    for filename in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, filename))

        points = tps_model.predict(image)
        # print(points)
        image = tps_model.transform(image, points, dst_size=(256, 32))
        tps_model.visual(image, None)
