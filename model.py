from detect.util import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
import numpy as np
import cv2
import copy
import recognize.predict_det as predict_det
from detect.turn import turnpicture
import recognize.utility as utility
import recognize.predict_cls as predict_cla
import recognize.predict_rec as predict_rec
from math import atan, degrees


class  OcrHandle(object):
    def __init__(self):
        self.args = utility.parse_args()
        self.args.cls_model_dir = r'.\recognize\ch_ppocr_mobile_v2.0_cls_infer/'
        self.args.rec_model_dir = r".\recognize\ch_PP-OCRv3_rec_infer/"   #路径，关gpu
        # self.args.rec_model_dir = r".\recognize\ch_ppocr_server_v2.0_rec_infer/"  # 路径，关gpu
        self.args.det_model_dir = r'.\recognize\ch_PP-OCRv3_det_infer/'
        self.args.use_gpu = True
        self.text_classifier = predict_cla.TextClassifier(self.args)
        self.text_recognizer = predict_rec.TextRecognizer(self.args)
        self.text_detect = predict_det.TextDetector(self.args)

        self.cixu = 0
    def clsandcrnnWithBox(self, im, boxes_list):
        boxes_list = sorted_boxes(np.array(boxes_list))                                   #box从上至下排序
        img_crop=[]
        for index, box in enumerate(boxes_list):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))         #将文字框投影变换，输入为im：图片、tmp_box：点坐标。并输出矩形图像partImg_array
            # IMG_OUT = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
            img_crop.append(partImg_array)

        img_list, cls_res, _ = self.text_classifier(img_crop)
        rec_res, elapse=self.text_recognizer(img_list)                                    #crnn文字识别，输出：rec_res为结果，elapse为置信度
        return rec_res,cls_res


    def crnnWithBox(self, im, boxes_list):
        boxes_list = sorted_boxes(np.array(boxes_list))                                   #box从上至下排序
        img_crop=[]
        for index, box in enumerate(boxes_list):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))         #将文字框投影变换，输入为im：图片、tmp_box：点坐标。并输出矩形图像partImg_array
            img_crop.append(partImg_array)
        rec_res, elapse=self.text_recognizer(img_crop)                                    #crnn文字识别，输出：rec_res为结果，elapse为置信度
        return rec_res

    def return_theta(self, img, theta):
        '''
        返回角度步骤：1.文本框检测 2.计算身份证号码文本框与label框角度 3.检测翻转是否倒置  返回2，3结果
        '''
        theta2 = 0
        if img.shape[0]>img.shape[1]:    #若高比宽长，则应旋转90
            img = turnpicture.rotate_image(img,90)
            theta2 = 90
        boxes_list, _ = self.text_detect(np.array(img))     #文本框检测
        if list(boxes_list)==[]:
            return [], theta, []
        img1 = img.copy()
        for i in range(len(boxes_list)):
            x1 = (boxes_list[i][0][0]-boxes_list[i][1][0])**2
            x2 = (boxes_list[i][0][1]-boxes_list[i][1][1])**2
            x3 = (boxes_list[i][2][0]-boxes_list[i][1][0])**2
            x4 = (boxes_list[i][2][1]-boxes_list[i][1][1])**2
            jud1 = (x1 + x2)**0.5
            jud2 = (x3 + x4)**0.5
            if jud2 > jud1 and jud2 > img.shape[0]:
                tan = (boxes_list[i][2][1]-boxes_list[i][1][1])/(boxes_list[i][2][0]-boxes_list[i][1][0])
                theta += degrees(atan(tan))
                break
            elif jud1> jud2 and jud1 > img.shape[0]:
                tan = (boxes_list[i][1][1] - boxes_list[i][0][1]) / (boxes_list[i][1][0] - boxes_list[i][0][0])
                theta += degrees(atan(tan))
                break                  #theta为角度
        pic = turnpicture.rotate_image(img1, theta)
        return theta + theta2
        # cv2.namedWindow('1',0)
        # cv2.circle(pic, (0, 6), 0, (0, 0, 255))
        # cv2.circle(pic, (143, 8), 0, (0, 0, 255))
        # cv2.circle(pic, (143, 35), 0, (0, 0, 255))
        # cv2.circle(pic, (0, 32), 0, (0, 0, 255))
        # cv2.imshow('1', pic)
        # cv2.waitKey(0)

    def return_cls_res(self, pic):
        boxes_list, _ = self.text_detect(np.array(pic))
        if boxes_list != []:
            boxes_list = sorted_boxes(np.array(boxes_list))  # box从上至下排序
            img_crop = []
            for index, box in enumerate(boxes_list):
                tmp_box = copy.deepcopy(box)
                partImg_array = get_rotate_crop_image(pic, tmp_box.astype(np.float32))
                # IMG_OUT = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
                img_crop.append(partImg_array)
            img_list, cls_res, _ = self.text_classifier(img_crop)
            return cls_res
        else:
            return []

    def predict_text(self,img,cls_res):
        img = turnpicture.rotate_image(img, cls_res)  # 根据图像是否倒置，决定旋转0°或180°
        boxes_list, _ = self.text_detect(np.array(img))
        rec_res = self.crnnWithBox(img, boxes_list)
        return rec_res

if __name__ == "__main__":
    pass
