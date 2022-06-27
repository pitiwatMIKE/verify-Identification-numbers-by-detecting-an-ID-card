import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

import cv2 
import numpy as np
from matplotlib import pyplot as plt

import pytesseract
import re
import socket
from init import *

HOST = 'localhost'
PORT = 5000

def send_data(id_):
    # สร้าง socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes(id_, 'utf-8'))

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

lst_id = list()
def find_data_max():
    global lst_id
    count_id = [lst_id.count(num_id) for num_id in lst_id]
    idx = count_id.index(max(count_id))
    return lst_id[idx]

def ocr_it(image, detections, detection_threshold):
    global lst_id
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]

        per = 26
        cut = region.shape[0]/100*per
        region_cut = region[:int(cut),:]

        #OCR
        text = pytesseract.image_to_string(region_cut)
        number_id_list = re.findall(r"(\d{1}) (\d{4}) (\d{5}) (\d{2}) (\d{1})",text)
        if number_id_list != []:
            number_id = "-".join(number_id_list[0])
            print(number_id)

            lst_id.append(number_id)
            if len(lst_id) == 3:
                number_id_ = find_data_max()
                send_data(number_id_)
                print("send: ", number_id_)
                lst_id = []
            # return number_id, region    

detection_threshold = 0.7
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    ocr_it(image_np_with_detections, detections, detection_threshold)
    # try:
    #     text, region = ocr_it(image_np_with_detections, detections, detection_threshold)
    #     # send_data(text)
    # except:
    #     pass

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break