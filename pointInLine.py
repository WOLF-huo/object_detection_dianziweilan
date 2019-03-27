# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:17:30 2019

@author: huoqi
"""

import cv2
import numpy as np

#image_path = r'rtsp://admin:QACXNT@10.0.1.106:554/h264/ch1/main/av_stream'
#gif = cv2.VideoCapture(image_path)

#gif.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
#gif.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[1]<poi[1]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后


def isPoiWithinPoly(poi,poly):
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

    #可以先判断点是否在外包矩形内 
    #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc=0 #交点个数
    for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly)):#[0,len-1]
            if i == (len(epoly)-1):
                s_poi=epoly[i]
                e_poi=epoly[0]
            else:
                s_poi=epoly[i]
                e_poi=epoly[i+1]
            if isRayIntersectsSegment(poi,s_poi,e_poi):
                sinsc+=1 #有交点就加1

    return True if sinsc%2==1 else  False


#size = (int(gif.get(cv2.CAP_PROP_FRAME_WIDTH)), int(gif.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#sizeStr = str(size[0]) + 'x' + str(size[1])
#print('sizeStr',sizeStr)

#pts = np.array([[148,162],[486,164],[462,413],[122,437],[72,272],[246,256]],np.int32)
#pts = pts.reshape(1,-1,2)
#
#_pts = np.array([[100,200],[100,300],[300,200],[300,400]],np.int32)
#_pts.reshape(-1,1,2)
#
#while True:
#    ret,frame = gif.read()
#    if ret == True:
#        frame = cv2.resize(frame,(1000, 600),interpolation=cv2.INTER_CUBIC)
#        cv2.polylines(frame,[pts],True,(0,255,255),2)
#        cv2.polylines(frame,[_pts],True,(0,255,255),2)
#        cv2.imshow('frame',frame)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#for pt in _pts:
#    result = isPoiWithinPoly(pt, pts)
#    if(result):
#        print(result,pt)
#        
#cv2.destroyAllWindows()



    