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

# @Time    : 2019/11/14 0014 20:09
# @Author  : peichao.xu
# @Email   : xj563853580@outlook.com
# @File    : server.py

# ==============================================================================

from __future__ import absolute_import, division, print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tornado.web
import tornado.options
import tornado.httpserver
import tornado.ioloop
# import signal

from tornado.options import define, options
from meter_ocr.web import OCRService


def server(port=6002):
    define('port', default=port, help="run on the given port", type=int)

    tornado.options.parse_command_line()

    app = tornado.web.Application(handlers=[
        (r"/predict", OCRService),
    ])

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port, address='0.0.0.0')
    # signal.signal(signal.SIGINT,
    #               tornado.ioloop.IOLoop.instance().add_callback_from_signal(tornado.ioloop.IOLoop.instance().stop()))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    server()
