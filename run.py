import argparse
import os
import sys
import time
from pathlib import Path
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolo.models.common import DetectMultiBackend
from yolo.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolo.utils.plots import Annotator, colors, save_one_box
from yolo.utils.torch_utils import select_device, time_sync
from detect.util  import sorted_boxes, get_rotate_crop_image
from detect.turn import turnpicture
from model import OcrHandle
from yolo.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

@torch.no_grad()
def run(
        weights=ROOT / r'runs\train\exp\weights\best.pt',  # model.pt path(s)
        source=ROOT / 'test',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'point.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    isFan = True

    source = str(source)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    timeinit = time.time()
    ocrhandle = OcrHandle()

    f = open(r'data.txt','r',encoding='utf-8')     #该文件为测试准确率的真实姓名，身份证号码
    data = f.readlines()
    pointFalse = []
    recFalse = []
    num = 0
    sum1 = 0
    sumname = 0
    sumname2 = 0
    sumhaoma = 0
    sumxb = 0
    for path, im, im0s, vid_cap, s in dataset:
        sum1 += 1
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        pred = model(im, augment=augment, visualize=False)       #yolo模型运行，第一次运行计算图像夹角
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        counts=0
        # Process predictions

        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            s += '%gx%g ' % im.shape[2:]  # print string
            points=[]
            jud = -1
            if len(det):
                theta = 0
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for index, c in enumerate(det[:, -1].unique()):
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "                   #s为输入照片信息，可删
                print(s)
                # Write results
                turn = turnpicture
                jud = turn.findrepeat(det)                    #针对yolo识别可能导致一个框同时属于两个类别的情况，进行后处理
                for index,(*xyxy, conf, cls) in enumerate(reversed(det)):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    if names[c]=='号码':
                        point1, point2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        theta= ocrhandle.return_theta(im0s[point1[1]:point2[1],point1[0]:point2[0]], theta)   #return_theta该函数通过检测身份证号码部分的夹角用以计算旋转角度

                ims = turn.rotate_image(im0s, theta)                            #针对识别角度旋转
                img = letterbox(ims, 640, stride=32, auto=True)[0]              #yolo训练前的预处理
                img = img.transpose((2, 0, 1))[::-1]
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if model.fp16 else img.float()
                img /= 255
                if len(img.shape) == 3:
                    img = img[None]
                pred = model(img, augment=augment, visualize=False)              #yolo第二次运行，用来寻找所需信息区域，pred为训练后的值
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dictlabelFan = {'姓名': '', '性别': '', '民族': '', '出生': '', '住址': '', '号码': ''}
                dictlabel = {'签发机关':'','有效期限':''}
                jud_cls = 1
                for i, det in enumerate(pred):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], ims.shape).round()
                    for index,(*xyxy, conf, cls) in enumerate(reversed(det)):
                        if jud!=-1 and index == jud:
                            continue
                        if isFan:
                            c = int(cls)
                            point1, point2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            if jud_cls == 1:
                                cls_res = ocrhandle.return_cls_res(ims[point1[1]:point2[1], point1[0]:point2[0]])
                                if cls_res==[]:
                                    continue
                                if cls_res[1]<0.6:
                                    continue
                                else:
                                    jud_cls = 0
                            if names[c] != '人脸' and names[c]!= '签发机关' and names[c]!= '有效期限':
                                result = ocrhandle.predict_text(ims[point1[1]:point2[1], point1[0]:point2[0]],int(cls_res[0]))
                                sum = ''
                                if result != []:
                                    for k in range(len(result)):
                                        dictlabelFan[names[c]] += result[k][0]
                            if names[c] == '人脸':                            #保存人脸信息，savepath为保存路径要修改
                                pathlist = path.split('\\')
                                savepath = 'C:\\Users\\86137\\Desktop\\renlian\\' + pathlist[-1]
                                cv2.imwrite(savepath, ims[point1[1]:point2[1], point1[0]:point2[0]])
                        else:
                            c = int(cls)
                            point1, point2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            if names[c] == '签发机关' or names[c] == '有效期限':


                                result = ocrhandle.predict_text(ims[point1[1]:point2[1], point1[0]:point2[0]],int(cls_res[0]))
                                sum = ''
                                if result != []:
                                    for k in range(len(result)):
                                        dictlabel[names[c]] += result[k][0]
                    '''
                    接口返回值：dictlabelFan为反面的识别数据；dictlabel为正面的识别数据
                    逻辑:当任意关键字为‘’，表示身份证有部分未成功识别，应当重新识别
                    '''
                    if isFan:
                        if '' in dictlabelFan.values():       #当有未识别的地方时，应当重新上传
                            print('有实体未识别出，识别有误')
                            print(dictlabelFan)
                        else:                                 #所有位置均能识别，返回dictlabelFan
                            print(dictlabelFan)

                    else:
                        if '' in dictlabel.values():
                            print('识别有误')
                            print(dictlabel)
                        else:
                            print(dictlabel)
                    '''
                    测试准确率
                    '''
                    if dictlabelFan['姓名'] == '' and dictlabelFan['号码'] == '' and dictlabelFan['性别'] == '':
                        sum1 = sum1 - 1
                        continue
                    imgname = path.split('\\')[-1]  # 获取imgname信息用来保存
                    key = True
                    for i in range(len(data)):
                        if imgname in data[i]:
                            data1 = data[i].strip().split('\t')
                            if dictlabelFan['姓名']!='' or dictlabelFan['号码']!='' or dictlabelFan['性别']!='':
                                if data1[0] == dictlabelFan['姓名'] and data1[2] == dictlabelFan['号码'] :
                                    if int(dictlabelFan['号码'][16])%2 == 1:
                                        dictlabelFan['性别']='男'
                                    elif int(dictlabelFan['号码'][16])%2 == 0:
                                        dictlabelFan['性别']='女'
                                    num += 1
                                    key = False
                                if data1[0]!=dictlabelFan['姓名']:
                                    if data1[2]!= dictlabelFan['号码']:
                                        sumname2 += 1
                                    sumname += 1
                                    cv2.imwrite('C:\\Users\\86137\\Desktop\\False\\xm\\' + imgname, im0)
                                if data1[2]!= dictlabelFan['号码']:
                                    sumhaoma+=1
                                    cv2.imwrite('C:\\Users\\86137\\Desktop\\False\\haoma\\' + imgname, im0)
                            if key:
                                print('姓名出错数量:',sumname,'号码出错数量:',sumhaoma,'姓名和号码都出错数量:',sumname2)
                                print('该身份证出错：',data1[0], dictlabelFan['姓名'],data1[1],dictlabelFan['性别'],data1[2], dictlabelFan['号码'])
    print('姓名出错数量:',sumname,'号码出错数量:',sumhaoma,'姓名和号码都出错数量:',sumname2)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'yolo\weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\86137\Desktop\pic', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'yolo/point.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--device', default='0', help='0 r cpu')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(imgPath):
     opt = parse_opt()
     check_requirements(exclude=('tensorboard', 'thop'))
     run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
