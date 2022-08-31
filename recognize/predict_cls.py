# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import math
import time
import traceback

import recognize.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list
from copy import deepcopy
logger = get_logger()


class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls', logger)
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        '''
        只取imglist[0]作为方向判断，方向值为0，180.
        由第一个文本段判断整体身份证照片是否需要180度旋转

        函数返回值：翻转后的imglist，推理出的角度，花费时间
        '''

        img_list = copy.deepcopy(img_list)
        # Calculate the aspect ratio of all text bars

        cls_res = [['', 0.0]]
        elapse = 0
        norm_img_batch = []
        max_wh_ratio = 0
        starttime = time.time()
        h, w = img_list[0].shape[0:2]
        wh_ratio = w * 1.0 / h
        norm_img = self.resize_norm_img(img_list[0])
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            prob_out = outputs[0]
        else:
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            prob_out = self.output_tensors[0].copy_to_cpu()
            self.predictor.try_shrink_memory()
        cls_result = self.postprocess_op(prob_out)
        elapse += time.time() - starttime

        img_all = deepcopy(img_list)
        label, score = cls_result[0]
        if '180' in label and score > self.cls_thresh:
            for i in range(len(img_list)):
                img_all[len(img_list)-1-i] = cv2.rotate(img_list[i], 1)
        return img_all, cls_result[0], elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_classifier = TextClassifier(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        if img:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))


if __name__ == "__main__":
    main(utility.parse_args())
