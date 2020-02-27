from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import sys
import time
import random
import pickle
from PIL import ImageFont, ImageDraw, Image
from pynput.mouse import Controller
import datetime



# setting
UNKNOWN_THRESHOLD = 0.7
CHECK_POINT = 5
WEB_SEND = True

# img_t = np.zeros((200,400,3),np.uint8)
fontpath = "../font/NanumGothicBold.ttf"
font = ImageFont.truetype(fontpath, 20)

print(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
matplotlib.use('agg')

def make_file(table):
    file_list = list()
    for key, val in table.items():
        if val[0] is True:
            file_list.append([key, 1, val[1]])
        else:
            file_list.append([key, 0, val[1]])
    output_name = '../src/web/입출현황.txt'
    with open(output_name, 'w') as f:
        for x in file_list:
            
            f.write(str(x[0])+" "+str(x[1])+ " " + str(x[2]) + '\n')
            print(x)
        print("file saved")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#print('Creating networks and loading parameters')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=True))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '../parameter/det/')

        minsize = 35  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        humans_dir = '../image/'
        humans_dir = facenet.get_dataset(humans_dir)
        HumanNames = []
        Human_hash = dict()
        Human_count = dict()
        human_len = len(humans_dir)

        for cls in humans_dir:
            HumanNames.append(cls.name)
            Human_hash[cls.name] = [False, 0]
            Human_count[cls.name] = 0

        if WEB_SEND:
            make_file(Human_hash)

        #print('Loading feature extractionodel')
        modeldir = '../parameter/20180402-114759/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = '../parameter/clf/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            #print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0
        if video_capture.isOpened() is False:
            print("camera is not connected")

        #wid = 640
        #hei = 480

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            # frame = cv2.resize(frame, (0,0), fx=1.5, fy=1.5)    #resize frame (optional)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_a = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                    exp_idx = 0

                    chk_name = []
                    tmp_arr = dict()
                    for cls in humans_dir:
                        tmp_arr[cls.name] = False

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        i -= exp_idx

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            exp_idx += 1
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                        scaled_a.append(facenet.prewhiten(scaled[i]))
                        scaled_reshape.append(scaled_a[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 10

                        if best_class_probabilities < UNKNOWN_THRESHOLD:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 5)  # boxing face

                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y), "외부인", font=font, fill=(0, 0, 255, 0))
                            frame = np.array(img_pil)


                            continue
 
                        result_names = HumanNames[best_class_indices[0]]
                        if Human_hash[result_names][0] is True:
                            attend_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
                            filename = "_".join([result_names])

                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 5)  # boxing face
                            frame[0:scaled[i].shape[0], 0:scaled[i].shape[1]] = scaled[i]
                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((0, scaled[i].shape[1] + 10), result_names + " 출석", font=font, fill=(0, 255, 0, 0))
                            frame = np.array(img_pil)
                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y), result_names + " " + str(np.round(best_class_probabilities, 2)), font=font, fill=(0, 255, 0, 0))
                            frame = np.array(img_pil)                                 

                            cv2.imwrite("../src/web/"+filename+".jpg", scaled[i])
                            time.sleep(0.2)
                        else:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 5)  # boxing face

                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((text_x, text_y), result_names + " " + str(np.round(best_class_probabilities, 2)), font=font, fill=(0, 0, 255, 0))
                            frame = np.array(img_pil)

                        print(result_names)


                        chk_name.append(result_names)
                    for x in chk_name:
                        tmp_arr[x] = True

                    for key, val in tmp_arr.items():
                        if Human_hash[key][0] is True:
                            continue
                        if val is True:
                            Human_count[key] += 1
                            if Human_count[key] == CHECK_POINT:
                                now = time.localtime()
                                s = "%04d/%02d/%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
                                Human_hash[key] = [True, s]
                                make_file(Human_hash)
                        else:
                            Human_count[key] = 0
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            str_t = 'FPS: %0.1f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            #cv2.putText(frame, str_t, (1566, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), thickness=2, lineType=2)
            # c+=1
            now = time.localtime()
            s = "%04d%02d%02d_%02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            #cv2.putText(frame, s, (1566, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), thickness=2, lineType=2)
            print("Current position: " + str(Controller().position))

            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
