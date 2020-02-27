# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   User
# @Last Modified time: 2019-10-30 22:06:32
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

#firbase update
cred = credentials.Certificate('./web_src/google-services.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
attend_doc = db.collection(u'Logs')

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
#predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

#error
threshold = 0.3

# load distance
with open("embeddings/embeddings.pkl", "rb") as f:
    (saved_embeds, names) = pickle.load(f)
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
with tf.Graph().as_default():
    with sess:

        saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
        saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        attend = {}
        prevTime = 0
        forder_path = "./faces/training/faceimage/"
        forder_list = os.listdir(forder_path)
        face_score_count = 0
        fontpath = "./font/NanumGothicBold.ttf"
        font = ImageFont.truetype(fontpath, 20)
        '''
        with open( './web_src/입출현황.txt') as data:
            lines = data.readlines()
            string_list = []
            for line in lines:
                string_list.append(line.rstrip())
        save_attend_text = './web_src/입출현황.txt'
        with open(save_attend_text, 'w') as f:
            for x in attend_tmp:
                print(x[0])
                if x[0] not in string_list:
                    f.write(str(x[0])+" "+str(x[1])+'\n')
                else:
                    pass
        '''
        while True:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            ret, frame = video_capture.read()

            now = time.localtime()
            attend_time = "%04d%02d%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            attend_times = "%02d:%02d:%02d" % (now.tm_hour, now.tm_min, now.tm_sec)
            attend_date = "%04d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday)

            #cv2.putText(frame, attend_time, (0, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            fps = 1/(sec)
            str = "FPS : %0.1f" % fps
            #cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            # preprocess faces
            h, w, _ = frame.shape
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            # detect faces
            confidences, boxes = ort_session.run(None, {input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            # locate faces
            faces = []
            boxes[boxes<0] = 0
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                aligned_face = cv2.resize(aligned_face, (112,112))
                aligned_face = aligned_face - 127.5
                aligned_face = aligned_face * 0.0078125
                faces.append(aligned_face)
            # face embedding
            if len(faces)>0:
                predictions = []
                faces = np.array(faces)
                feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                embeds = sess.run(embeddings, feed_dict=feed_dict)
                # prediciton using distance
                for embedding in embeds:
                    diff = np.subtract(saved_embeds, embedding)
                    dist = np.sum(np.square(diff), 1)
                    idx = np.argmin(dist)
                    if dist[idx] < threshold:
                        predictions.append(names[idx])
                        if names[idx] not in attend and names[idx] in forder_list :
                            face_score_count += 1
                            if face_score_count > 5:
                                attend[names[idx]] = attend_time
                                user_id = db.collection(u'User').where(u'name',u'==', names[idx]).stream()
                                for i in user_id:
                                    id = (i.to_dict()['uid'])
                                    attend_doc.document(id).collection(u'ai').document(attend_date).set({
                                        u'time': attend_times
                                    })
                            else:
                                pass
                        else:
                            face_score_count = 0
                            pass
                    else:
                        predictions.append("외부인")
                print(attend)
                # draw
                for i in range(boxes.shape[0]):
                    box = boxes[i, :]
                    text = f"{predictions[i]}"
                    x1, y1, x2, y2 = box
                    #draw a label with a name below the face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x1 - 6, y2 - 6), text, font=font, fill=(0, 255, 0, 0))
                    frame = np.array(img_pil)
            cv2.imshow('Video', frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()