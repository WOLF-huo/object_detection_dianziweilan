# -*- coding: utf-8 -*-
"""
Created on :2018/8/29 18:28
@author: sgs4176
"""
import os
import cv2
import time
#import argparse
#import multiprocessing
import numpy as np
import tensorflow as tf
import sys
#import subprocess as sp
#import psycopg2
#import configparser
#import datetime

#from utils.app_utils import FPS, WebcamVideoStream
#from multiprocessing import Queue, Pool

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import pointInLine as pl

# # This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")


# What model to download.
#MODEL_NAME = 'frozen_inference_graph_ssd_m_v1'
#MODEL_NAME = 'frozen_inference_graph_ssd_m_v2'
#MODEL_NAME = 'frozen_inference_graph_frcnn_inception'
MODEL_NAME = './object_detection/ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('./object_detection/data', 'mscoco_label_person.pbtxt')

# pgsql config
#PATH_TO_PGSQL = os.path.join('data', 'pgsql.config')

# rtmp address
#rtmpUrl = 'rtmp://10.0.1.63:1935/live/stream'
 
NUM_CLASSES = 1

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#category_index[1]['name']=u'安全帽'
#category_index[2]['name']=u'无安全帽'

# connect to pgsql 
#def connect_to_pgsql(path):
#    cf = configparser.ConfigParser()
#    cf.read(path)
#    host = cf.get('db', 'db_host')
#    port = cf.get('db', 'db_port')
#    database = cf.get('db', 'db_database')
#    user = cf.get('db', 'db_user')
#    password = cf.get('db', 'db_password')    
#    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
#    return conn


# put result to rtmp streaming
#def put_to_rtmp(rtmpUrl, sizeStr, fps_num):
#    command = ['ffmpeg',
#                '-y',
#                '-f', 'rawvideo',
#                '-vcodec','rawvideo',
#                '-pix_fmt', 'bgr24',
#                '-s', sizeStr,
#                '-r', str(fps_num),
#                '-i', '-',
#                '-c:v', 'libx264',
#                '-pix_fmt', 'yuv420p',
#                '-preset', 'ultrafast',
#                '-f', 'flv',
#                rtmpUrl]
#    proc = sp.Popen(command, stdin=sp.PIPE, shell=True) #shell=
#    return proc


# format detect result
#def detect_result_to_pgsql(scores, classes, category_index, video_id, conn):
#    classes = np.squeeze(classes)
#    scores  = np.squeeze(scores)
#    cur = conn.cursor()
#    
#    num = scores[scores>0.8].shape[0]
#    class_id = classes[scores>0.8]
#    for i in range(num):
#        if class_id[i] == 2:
#            timeStamp = time.time()
#            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#            try:
#                cur.execute("insert into t_mon_message VALUES (%s, %s, '0', %s, %s);" , (int(round(timeStamp * 1000)), str(class_id[i]), nowTime, video_id))
#            except Exception as e:
#                print('excute sql error!')
#    conn.commit()
#    cur.close()




#GPU特有   
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.allow_soft_placement=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.7

if __name__ == '__main__':
    pts = np.array([[800,100],[1000,100],[1000,600],[800,600]],np.int32)
    pts = pts.reshape(1,-1,2)

    cap = cv2.VideoCapture(r'rtsp://admin:QACXNT@10.0.1.106:554/h264/ch1/main/av_stream')
#    cap = cv2.VideoCapture(r'rtmp://rtmp.open.ys7.com/openlive/f01c46bc25e3425390a6965838e7cdfe')
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(size[0]) + 'x' + str(size[1])
    fps = cap.get(cv2.CAP_PROP_FPS) # 30p/self
    if fps == 0:
        print('摄像头未开启！')
        raise NameError
    elif fps < 100:
        fps_num = int(fps)
    else:
        fps_num = int(fps/10000)
    _numFrames = 0
    
    # rtmp video streaming
#    proc = put_to_rtmp(rtmpUrl, sizeStr ,fps_num)
    
#    try:
#        conn = connect_to_pgsql(PATH_TO_PGSQL)
#        print('Connect to PGSQL success!')
#    except Exception as e:
#        print('PGSQL connect error:',e) 
        
    with detection_graph.as_default():
#        with tf.Session(graph=detection_graph, config=config) as sess:
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            while True:
                ret,image = cap.read()
                t = time.time()
                if ret == True:
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    final_score = np.squeeze(scores)    
                    count = 0
                    for j in range(100):
                        if scores is None or final_score[j] > 0.5:
                            count = count + 1
                    print("the count of objects is: ", count )
                    #(im_width, im_height) = image.size
                    max_boxes_to_draw = boxes.shape[0]
                    _scores = np.zeros(final_score.shape[0])
                    for i in range(count):
                        y_min = boxes[0][i][0]*im_height
                        x_min = boxes[0][i][1]*im_width
                        y_max = boxes[0][i][2]*im_height
                        x_max = boxes[0][i][3]*im_width
#                        print("object{0}: {1}".format(i,category_index[classes[0][i]]['name']),
#                                         ',Center_X:',int((x_min+x_max)/2),',Center_Y:',int((y_min+y_max)/2))
#                    y_min = int(boxes[0][0][0]*im_height)
#                    x_min = int(boxes[0][0][1]*im_width)
#                    y_max = int(boxes[0][0][2]*im_height)
#                    x_max = int(boxes[0][0][3]*im_width)
#                    print(x_min,y_min,x_max,y_max)                                  
                        _pts = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]],np.int32)
                        for pt in _pts:
                            result = pl.isPoiWithinPoly(pt, pts)
                            if(result):
                                _scores[i] = final_score[i]
                                break
                    for m in range(max_boxes_to_draw):
                        class_ = np.squeeze(classes)[m]
                        if np.int16(class_) == 1:                            
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                _scores,
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=2)
                        else:
                            continue
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#                    proc.stdin.write(image_np.tobytes())
                    cv2.polylines(image_np,[pts],True,(0,255,255),2)
                    cv2.imshow("frame",image_np)
#                    if (_numFrames%50 == 0):
#                        detect_result_to_pgsql(scores, classes, category_index, '1', conn)
#                    _numFrames += 1    
                    print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print('结束时间：',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                    cap = cv2.VideoCapture(r'rtsp://admin:QACXNT@10.0.1.106:554/h264/ch1/main/av_stream')
#            conn.close()
#        proc.stdin.close()
        cap.release()
        cv2.destroyAllWindows()