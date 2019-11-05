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
# @File    : crnn.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import tensorflow as tf


class RNN(object):
    def __init__(self, weights, allow_growth=True, allow_soft_placement=True, log_device_placement=False):
        session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                        log_device_placement=log_device_placement)
        session_config.gpu_options.allow_growth = allow_growth

        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.input_size = (320, 32)

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=session_config)

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with open(weights, "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

    def decode(self, indices, values, batchsize):
        '''
        :param indices: tf.SparseTensor.indices
        :param values: tf.SparseTensor.values
        :param batchsize:
        :return:
        '''
        decoded_indexes = {}

        for i, idx in enumerate(indices):
            if idx[0] not in decoded_indexes:
                decoded_indexes[idx[0]] = []
            decoded_indexes[idx[0]].append(values[i])

        mi = len(self.chars)
        for k, v in decoded_indexes.items():
            decoded_indexes[k] = ''.join([self.chars[i] if i < mi else ' {} '.format(str(i)) for i in v])

        if len(decoded_indexes) < batchsize:
            for i in range(len(decoded_indexes), batchsize):
                decoded_indexes[i] = ''

        if len(decoded_indexes) > 0:
            max_idx = max(decoded_indexes.keys()) + 1
            words = ['' for _ in range(max_idx)]
            for k, v in decoded_indexes.items():
                words[k] = v
            return words
        else:
            return []

    def _preprocess(self, img, size, interp=cv2.INTER_CUBIC):
        h, w, _ = img.shape

        nw = int(np.ceil(size[1] * w / h))

        if not size[0] is None:
            nw = size[0] if nw > size[0] else nw

        resized = cv2.resize(img, (nw, size[1]), interpolation=interp)

        if size[0] is None:
            maxw = nw if nw % size[1] == 0 else (nw // size[1] + 1) * size[1]
        else:
            assert size[0] % size[1] == 0, size[0]
            maxw = size[0]

        temp = np.zeros((size[1], size[0], 3), dtype=img.dtype)
        exp = (maxw - nw) // 2
        temp[:, exp:exp + nw, :] = resized

        temp = temp.astype('float32')
        temp /= 127.5
        temp -= 1.

        return temp

    def predict(self, img):
        image = self._preprocess(img, size=self.input_size)
        indices, values = self.session.run(["indices:0", "values:0"], feed_dict={"inputs:0": [image]})
        text = self.decode(indices, values, 1)
        return text


if __name__ == '__main__':
    rnn = RNN('/code/disk1/xupeichao/data/dianbiao/data_v1/crnn/v1/exports/model_76000.pb')
    image = cv2.imread(
        '/code/disk1/xupeichao/data/dianbiao/data_v1/croped/images/300f2312-408c-4eb6-b9dc-f4c5b09514ff_0000.jpg')
    text = rnn.predict(image)

    print(text)
