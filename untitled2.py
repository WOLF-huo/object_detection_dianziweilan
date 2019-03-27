# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:36:21 2019

@author: huoqi
"""

import cv2
import numpy as np
coor_x,coor_y = -1, -1 # 初始值并无意义，只是为了能够使用np.row_stack函数

"""定义视频编码器
FourCC全称Four-Character Codes，代表四字符代码 (four character code), 
它是一个32位的标示符，其实就是typedef unsigned int FOURCC;
是一种独立标示视频数据流格式的四字符代码。
因此cv2.VideoWriter_fourcc()函数的作用是输入四个字符代码即可得到对应的视频编码器。
"""
fourcc = cv2.VideoWriter_fourcc(*'XVID') # 使用XVID编码器
camera = cv2.VideoCapture(r'rtsp://admin:QACXNT@10.0.1.106:554/h264/ch1/main/av_stream') # 从文件读取视频,Todo:只需要修改成自己的视频路径即可进行测试
fps = camera.get(cv2.CAP_PROP_FPS)# 获取视频帧率
print('视频帧率：%d fps' %fps)

# 判断视频是否成功打开
if (camera.isOpened()):
    print('视频已打开')
else:
    print('视频打开失败!')

# # 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
       int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print ('视频尺寸:'+repr(size))
coor = np.array([[1,1]]) # Todo:初始值并无意义，只是为了能够使用np.row_stack函数

def OnMouseAction(event,x,y,flags,param):
    global coor_x,coor_y,coor
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        print("%s" %x,y)
        coor_x ,coor_y = x ,y
        coor_m = [coor_x,coor_y]
        coor = np.row_stack((coor,coor_m))
    elif event==cv2.EVENT_LBUTTONUP:
        cv2.line(img, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)
    elif event==cv2.EVENT_RBUTTONDOWN :
        print("右键点击")
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳")
    elif event==cv2.EVENT_MBUTTONDOWN :
        print("中键点击")
'''
创建回调函数的函数setMouseCallback()；
下面把回调函数与OpenCV窗口绑定在一起
'''
grabbed, img = camera.read() # 逐帧采集视频流
cv2.namedWindow('Image')
cv2.setMouseCallback('Image',OnMouseAction)
while(1):
    cv2.imshow('Image',img)
    k=cv2.waitKey(1) & 0xFF
    if k==ord(' '): # 空格退出操作
        break
cv2.destroyAllWindows() # 关闭页面

Width_choose = coor[2,0]-coor[1,0] # 选中区域的宽
Height_choose = coor[2, 1] - coor[1, 1] # 选中区域的高
print("视频选中区域的宽：%d" %Width_choose,'\n'"视频选中区域的高：%d" %Height_choose)
out = cv2.VideoWriter('output_test1.avi',fourcc, fps, (Width_choose,Height_choose)) # 参数分别是：保存的文件名、编码器、帧率、视频宽高
Video_choose = np.zeros((Width_choose, Height_choose, 3), np.uint8)

while True:
    grabbed, frame = camera.read() # 逐帧采集视频流
    if not grabbed:
        break
    gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转灰度图
    frame_data = np.array(gray_lwpCV)  # 每一帧循环存入数组
    box_data = frame_data[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]] # 取矩形目标区域
    pixel_sum = np.sum(box_data, axis=1) # 行求和q
    x = range(Height_choose)
    emptyImage = np.zeros((Width_choose * 10, Height_choose * 2, 3), np.uint8)
    Video_choose = frame[coor[1,1]:coor[2,1],coor[1,0]:coor[2,0]]
    out.write(Video_choose)
    cv2.imshow('Video_choose', Video_choose)
    for i in x:
        cv2.rectangle(emptyImage, (i*2, (Width_choose-pixel_sum[i]//255)*10), ((i+1)*2, Width_choose*10), (255, 0, 0), 1)
    emptyImage = cv2.resize(emptyImage, (320, 240))
    lwpCV_box = cv2.rectangle(frame, tuple(coor[1,:]), tuple(coor[2,:]), (0, 255, 0), 2)

    cv2.imshow('lwpCVWindow', frame) # 显示采集到的视频流
    cv2.imshow('sum', emptyImage)  # 显示画出的条形图
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
out.release()
camera.release()
cv2.destroyAllWindows()