
# 依赖python环境

+ python3.5

# 安装说明

执行即可 `pip install MeterOCR-0.0.9-cp35-cp35m-win_amd64.whl`

# 依赖的pip包

+ cv2 (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python)
+ matplotlib (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib)
+ tensorflow==1.6 (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.6)



# api

## 离线调用接口：

```python
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
        img: cv2.Mat 格式图片
        has_det: 是否开启检测模式(适用于大图)，小图请设置为False

        Usage::
            >>> from meter_ocr.ifs import Interface
            >>> elec_model = MeterElecOCR()
            >>> result = elec_model.predict(cv2.UMat)
        """
        return self._predict(img, has_det=True)
```

## 在线调用接口

### POST请求 

`localhost:6002/predict?image=b'abc';has_det='false'`

|名称|类型|必填|说明|
| --- | --- | --- | --- |
|image|base64|True|以base64编码的jpg格式图片|
|has_det|string|False|参数为'True'或'true'时，适用于大图大图识别，否则适用于裁剪后的小图识别|

### 返回参数说明

|名称|类型|说明|
| --- | --- | --- |
|code|int|错误码，非0则返回异常|
|ftime|时间|请求时间戳|
|message|string|错误信息|
|result|字典|包含识别结果|
|time|float|识别耗时|
|bndbox|1d-数组|存储检测区域坐标，格式为：[左上x,左上y,右下x,右下y]|
|polygon|2d-数组|存储斜框区域坐标，格式为：[[左上], [右上], [右下], [左下]]|
|text|string|识别的文本结果|

### JSON 返回示例
```json
{
    "code": 0,
    "ftime": "2019-11-18 21:47:17 PM 1",
    "message": "success",
    "result": {
        "time": 0.031061649322509766,
        "bndbox": [
            59,
            14,
            242,
            50,
            0.9962421655654907
        ],
        "polygon": [
            [
                85.4101791381836,
                56.29902648925781
            ],
            [
                258.4898681640625,
                71.1941146850586
            ],
            [
                258.367431640625,
                91.19723510742188
            ],
            [
                84.28414154052734,
                80.03681945800781
            ]
        ],
        "text": "01673227"
    }
}
```