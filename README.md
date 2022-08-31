# 身份证识别
## 代码详解：
https://zhuanlan.zhihu.com/p/559978244
## 使用方法：
运行 run.py。 
### requiremnets
matplotlib>=3.2.2 
numpy>=1.18.5 
opencv-python>=4.1.1  
Pillow>=7.1.2 
PyYAML>=5.3.1 
requests>=2.23.0  
scipy>=1.4.1  # Google Colab version  
torch>=1.7.0  
torchvision>=0.8.1  
tqdm>=4.41.0  
protobuf<=3.20.1  # https://github.com/ultralytics/yolov5/issues/8012 
imgaug==0.4.0 

### 使用的cuda
cuda10.2  
cudnn8.4.1  
显卡2070  

### 参数修改：
48行  选择图片是正面或反面  ifFan = True表示图片为反面：人脸面  
217行  修改图片路径 default= 图片路径或文件夹路径  
152行 保存人脸图片，保存路径是savepath，ims[point1[1]:point2[1], point1[0]:point2[0]]是应存图片  
167-183行 ocr识别返回值，返回重新识别或者识别结果  
184行-214行 测试图片准确率。应把217行路径改为文件夹路径 
