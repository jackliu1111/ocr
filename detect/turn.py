import math
import cv2
import os
import numpy as np
from math import *
import cv2
import numpy as np


class turnpicture():
    # -*- coding:utf-8 -*-

    def rotate_bound(image, angle):
        '''
         . 旋转图片
         . @param image    opencv读取后的图像
         . @param angle    (逆)旋转角度
        '''

        # img = cv2.imread("img/1.jpg")
        (h, w) = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        # 计算图像的新边界维数
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv2.warpAffine(image, M, (newW, newH))  # borderValue 缺省，默认是黑色

    def rotate_image(image, angle):
        '''
         . 旋转图片
         . @param image    opencv读取后的图像
         . @param angle    (逆)旋转角度
        '''

        h, w = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
        newW = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        newH = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        return cv2.warpAffine(image, M, (newW, newH), borderValue=(255, 255, 255))


    def returnPoints(points):
        '''
        该函数将坐标点分为左上,右上，右下，左下，顺时针输出
        以x最小值点为左上点，另外三点中随机一点连线，若另外两点分别再斜率大于小于0，则确定四点方向。
        '''
        sortpoint = sorted(range(len(points)), key=lambda k:points[k][0],reverse=False)
        for i in range(4):
            sortpoint[i] = points[sortpoint[i]]
        x0,y0 = sortpoint[0]
        for index,[x1,y1] in enumerate(sortpoint[1:]):

            if x1==x0:
                continue

            a = (y1-y0)/(x1-x0)
            b = y1-a*x1
            if index==0:
                distence1 = a * sortpoint[2][0] + b - sortpoint[2][1]
                distence2 = a * sortpoint[3][0] + b - sortpoint[3][1]
                if distence2*distence1<0:
                    if distence1>0:
                        leftup ,rightup ,rightdown, leftdown=sortpoint[0], sortpoint[2],sortpoint[1],sortpoint[3]
                    else:
                        leftup ,rightup ,rightdown, leftdown=sortpoint[0], sortpoint[3],sortpoint[1],sortpoint[2]
                    break
            if index == 1:
                distence1 = a * sortpoint[1][0] + b - sortpoint[1][1]
                distence2 = a * sortpoint[3][0] + b - sortpoint[3][1]
                if distence2*distence1<0:
                    if distence1>0:
                        leftup, rightup, rightdown, leftdown = sortpoint[0], sortpoint[1], sortpoint[2], sortpoint[3]
                    else:
                        leftup, rightup, rightdown, leftdown = sortpoint[0], sortpoint[3], sortpoint[2], sortpoint[1]
                    break
            if index == 2:
                distence1 = a * sortpoint[1][0] + b - sortpoint[1][1]
                distence2 = a * sortpoint[2][0] + b - sortpoint[2][1]
                if distence2*distence1<0:
                    if distence1>0:
                        leftup, rightup, rightdown, leftdown = sortpoint[0], sortpoint[1], sortpoint[3], sortpoint[2]
                    else:
                        leftup, rightup, rightdown, leftdown = sortpoint[0], sortpoint[2], sortpoint[3], sortpoint[1]
                    break
        points = [leftup, rightup, rightdown, leftdown]
        return points

    def findrepeat(det):
        '''
        算法思路：针对det坐标框
        首先 遍历找到重复的类别，jud为重复的类
        第二步 针对每一个det识别后面的所有框,实现所有框重叠范围的比较
        第三步 一旦两个框重叠，根据第一步的类别确定哪个框需删除，并返回该框次序
        '''
        labellist = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        jud = 0
        for index, (*xyxy, conf, cls) in enumerate(reversed(det)):
            labellist[int(cls)] += 1
            if labellist[int(cls)] > 1:
                jud = int(cls)
        if jud:
            for index, (*xyxy, conf, cls) in enumerate(reversed(det)):
                for j in range(index + 1, len(det)):
                    xyxy2 = reversed(det)[j][0:4]
                    conf2 = reversed(det)[j][4]
                    cls2 = reversed(det)[j][5]
                    if cls2 != cls:
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        x3, y3, x4, y4 = int(xyxy2[0]), int(xyxy2[1]), int(xyxy2[2]), int(xyxy2[3])
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        x3, x4 = min(x3, x4), max(x3, x4)
                        y3, y4 = min(y3, y4), max(y3, y4)
                        if (x2 <= x3 or x4 <= x1) and (y2 <= y3 or y4 <= y1):
                            pass
                        else:
                            lens = min(x2, x4) - max(x1, x3)
                            wide = min(y2, y4) - max(y1, y3)
                            lensmax = max(x2, x4) - min(x1, x3)
                            lensmin = max(y2, y4) - min(y1, y3)
                            if (lens * wide) / (lensmax * lensmin) > 0.5 and cls == jud:
                                return index
                            elif (lens * wide) / (lensmax * lensmin) > 0.5 and cls2 == jud:
                                return j
        else:
            return -1