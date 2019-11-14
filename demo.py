#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/3 0003 18:11
# @Author  : none
# @Email   : none
# @File    : demo.py
# ==============================================================================

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from meter_ocr.ifs import Interface


class MeterElecOCR(Interface):
    """A user-created :class:`MeterElecOCR <Interface>` object.

    继承识别类

    Usage::
        >>> from meter_ocr.ifs import Interface
        >>> elec_model = MeterElecOCR()
    """

    def __init__(self):
        super(MeterElecOCR, self).__init__()
        print(self.message)

    def predict(self, img):
        """主要调用接口
        识别待检测图片，并返回字典结果

        Usage::
            >>> from meter_ocr.ifs import Interface
            >>> elec_model = MeterElecOCR()
            >>> result = elec_model.predict(cv2.UMat)
        """
        return self._predict(img)

    def vis_demo(self, image):
        """接口调用演示，可自定义调试

        Usage::
            >>> from meter_ocr.ifs import Interface
            >>> elec_model = MeterElecOCR()
            >>> elec_model.vis_demo(image)
        """
        # 检测识别区域
        boxes = self.det_model.predict(image)  # boxes: [[x0, y0, x1, y1, cls, score]], shape: (n, 6)

        # { 可视化
        plt.subplot(221)
        self.det_model.visual(image.copy(), boxes, save=False, show=False)
        # }

        if len(boxes) > 0:
            x0, y0, x1, y1, cls, score = boxes[0]
            roi = self.crop_area(image, [x0, y0, x1, y1], scale=1.25)

            # 校正待识别区域
            points = self.tps_model.predict(roi)  # points: [lt, rt, rb, lb], shape: (4, 2)

            # { 可视化
            plt.subplot(222)
            self.tps_model.visual(roi.copy(), points, save=False, show=False)
            # }

            tfm_img = self.tps_model.transform(roi, points, dst_size=(320, 32))

            # { 可视化
            plt.subplot(223)
            self.tps_model.visual(tfm_img.copy(), None, save=False, show=False)
            # }

            # 识别文字内容
            text = self.rnn_model.predict(tfm_img)

            # { 可视化
            plt.title(text[0])
        plt.show()
        # }


if __name__ == '__main__':
    import os
    import cv2

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    elec_model = MeterElecOCR()

    image_dir = 'images'
    for filename in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, filename))
        elec_model.vis_demo(image)

        result = elec_model.predict(image)
        print(result)
