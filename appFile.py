import numpy as np

import json
import datetime
import base64
import zmq
import cv2
import time
import dlib
import imutils
import logging
import argparse
import requests
import io

import global_config
import helper
from ultralytics import YOLO


from matplotlib import pyplot as plt
from geti_sdk.deployment import Deployment
from imutils.video import FPS
from tracker.trackableobject import TrackableObject
from tracker.centroidtracker import CentroidTracker
# from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)


#### =================== MAIN FUNCTION =================== ####

def main(video_path, offline_deployment, sensor_id):
    ## ================= Config Devices =================
    daqID = helper.loadConfig(sensor_id)["config_device"]["daqID"]
    camID = helper.loadConfig(sensor_id)["config_device"]["camID"]
    
    ## ================= Config Local =================
    verbose = helper.loadConfig(sensor_id)["config_local"]["verbose"]
    confidence_val = helper.loadConfig(sensor_id)["config_local"]["confidence_val"]
    skip_frame = helper.loadConfig(sensor_id)["config_local"]["skip_frame"]
    counting = helper.loadConfig(sensor_id)["config_local"]["counting"]
    
    ## ================= Config Parkir Liar =================
    parkir_liar_detection = helper.loadConfig(sensor_id)["config_parkir_liar"]["parkir_liar_detection"]
    polygon_roi = helper.loadConfig(sensor_id)["config_parkir_liar"]["polygon_roi"]
    parkir_liar_object_count_limit = helper.loadConfig(sensor_id)["config_parkir_liar"]["object_count_limit"]
    parkir_liar_object_time_limit = helper.loadConfig(sensor_id)["config_parkir_liar"]["object_time_limit"]

    ## ================= Config Object Mendekati Object =================
    object_mendekati_object_detection = helper.loadConfig(sensor_id)["config_object_mendekati_object"]["object_mendekati_object_detection"]
    distance_threshold = helper.loadConfig(sensor_id)["config_object_mendekati_object"]["distance_threshold"]
    object_label = helper.loadConfig(sensor_id)["config_object_mendekati_object"]["object_label"]
    motor_label = helper.loadConfig(sensor_id)["config_object_mendekati_object"]["motor_label"]
    orang_label = helper.loadConfig(sensor_id)["config_object_mendekati_object"]["orang_label"]
    
    ## ================= Config Demonstrasi Object ================= ##
    demonstrasi_detection = helper.loadConfig(sensor_id)["config_demonstrasi"]["demonstrasi_detection"]
    demonstrasi_vehicle_limit = helper.loadConfig(sensor_id)["config_demonstrasi"]["demonstrasi_vehicle_limit"]
    demonstrasi_bike_limit = helper.loadConfig(sensor_id)["config_demonstrasi"]["demonstrasi_bike_limit"]
    demonstrasi_people_limit = helper.loadConfig(sensor_id)["config_demonstrasi"]["demonstrasi_people_limit"]
    
    ## ================= Config Pencurian Object ================= ##
    pencurian_motor_detection = helper.loadConfig(sensor_id)["config_pencurian_motor"]["pencurian_motor_detection"]
    pencurian_motor_time_limit = helper.loadConfig(sensor_id)["config_pencurian_motor"]["object_time_limit"]
    polygon_roi_pencurian_motor = helper.loadConfig(sensor_id)["config_pencurian_motor"]["polygon_roi_pencurian_motor"]
    
    ## ================= Config Gerakan Mencurigakan Object ================= ##
    gerakan_mencurigakan_detection = helper.loadConfig(sensor_id)["config_gerakan_mencurigakan"]["gerakan_mencurigakan_detection"]
    gerakan_mencurigakan_time_limit = helper.loadConfig(sensor_id)["config_gerakan_mencurigakan"]["object_time_limit"]
    
    # ## ================= Config Objek Tertinggal Object ================= ##
    # objek_tertinggal_detection = helper.loadConfig(sensor_id)["config_barang_tertinggal"]["gerakan_mencurigakan_detection"]
    # objek_tertinggal_time_limit = helper.loadConfig(sensor_id)["config_barang_tertinggal"]["object_time_limit"]
    
    ## ================= Config PKL Object ================= ##
    pkl_detection = helper.loadConfig(sensor_id)["config_pkl"]["pkl_detection"]
    pkl_time_limit = helper.loadConfig(sensor_id)["config_pkl"]["object_time_limit"]
    polygon_roi_pkl = helper.loadConfig(sensor_id)["config_pkl"]["polygon_roi_pkl"]

    ## ================= Config Pencurian Motor ================= ##
    pencurian_motor_detection = helper.loadConfig(sensor_id)["config_pencurian_motor"]["pencurian_motor_detection"]
    pencurian_motor_stillness_threshold = helper.loadConfig(sensor_id)["config_pencurian_motor"]["stillness_threshold"]
    pencurian_motor_roi = helper.loadConfig(sensor_id)["config_pencurian_motor"]["polygon_roi_pencurian_motor"]

    ## ================= Config API =================
    zmq_address = helper.loadConfig(sensor_id)["config_api"]["zmq_address"]
    serviceStandalone = helper.loadConfig(sensor_id)["config_api"]["urlService"]
    DiamBergerakIDs = []

    def movingCallback(id, name, centroid, roi, bbox, frame):
        points = np.array(roi, np.int32)
        points = points.reshape((-1, 1, 2))
        
        point_to_check = (int(centroid[0]), int(centroid[1]))
        result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)

        _, buffer = cv2.imencode('.jpg', frame)

        # Konversi buffer ke bytes
        image_bytes = buffer.tobytes()

        # Encode bytes ke base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        imageName = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}.jpg"
        upload_success = helper.uploadFile(image_base64, imageName)
        
        if result > 0 and upload_success:
            data_result = {
                "colorRecognition" : False,
                "alpr" : False,
                "objectData" : name,
                "objectID" : 0,
                "state" : 1,
                "bbox": bbox.tolist(),
                "context" : f"{name.lower()}-bergerak-ROI-parkiran",
                "class" : "person" if name == "Orang" else "vehicle",
                "daqID" : daqID,
                "camID" : camID,
                "filename": imageName
            }

            # KIRIM DATA
            if id not in DiamBergerakIDs:
                helper.send_mqtt(data_result)
                DiamBergerakIDs.append(id)
                logger.info("Pencurian motor Send")



    logger.info("Starting the video..") if verbose else None

    vs = cv2.VideoCapture(video_path)

    W = H = None

    ct = CentroidTracker(maxDisappeared=500, maxDistance=500)

    trackers = []
    labels = []
    object_id = []

    trackableObjects = {}

    totalFrames = 0

    fps = FPS().start()

    context = zmq.Context()
    footage_socket = context.socket(zmq.PUB)
    footage_socket.bind(zmq_address)


    ## ==================== PEOPLE COUNTING SECTION ==================== ##
    if counting:
        trackers = []
    ## ==================== PEOPLE COUNTING SECTION ==================== ##

    ## ==================== OBJECT MENDEKATI OBJECT SECTION ==================== ##
    if object_mendekati_object_detection:
        lastSend = 0
        lastObjectMendekatiObject = ""
    ## ==================== OBJECT MENDEKATI OBJECT SECTION ==================== #
    
    ## ==================== DEMONSTRASI SECTION ==================== ##
    if demonstrasi_detection :
        people_counted = []
        vehicle_counted = []
        bike_counted = []
        count_frame_det_demonstrasi = 0
        start_time_demonstrasi = None
    ## ==================== DEMONSTRASI SECTION ==================== ##
    
    ## ==================== PARKIR LIAR SECTION ==================== ##
    if parkir_liar_detection :
        _elapsed_time_obj_parkir_liar = None
    ## ==================== PARKIR LIAR SECTION ==================== ##
    
    ## ==================== PKL SECTION ==================== ##
    if pkl_detection :
        _elapsed_time_obj_pkl = None
    ## ==================== PKL SECTION ==================== ##
    
    ## ==================== Gerakan Mencurigakan SECTION ==================== ##
    if pencurian_motor_detection :
        _elapsed_time_obj_pencurian_motor = None
    ## ==================== Gerakan Mencurigakan SECTION ==================== ##

    messagesQueue = []
    imageName = None

    # Define video properties
    output_file = 'output.mp4'  # Output video file name
    frame_width = 1280          # Width of the video
    frame_height = 720          # Height of the video
    fpsRate = 15                    # Frames per second

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec (e.g., 'XVID', 'MJPG', 'MP4V')
    # out = cv2.VideoWriter(output_file, fourcc, fpsRate, (frame_width, frame_height))

    
    while True:

        ## ==================== OBJECT MENDEKATI OBJECT SECTION ==================== ##
        if object_mendekati_object_detection:
            objectIDs = []
        ## ==================== OBJECT MENDEKATI OBJECT SECTION ==================== #

        hasFrame, frame = vs.read()
        
        if frame is not None and frame.size > 0:
            frame = frame
        else:
            print("Warning: Frame is empty or invalid.")
        
        imageName = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}.jpg"
        
        if not hasFrame or frame is None:
            frame = np.zeros((1020, 480, 3), dtype=np.uint8)
            vs.release()
            vs = cv2.VideoCapture(video_path)
            logger.info('CAM ERROR') if verbose else None
        debugFrame = frame.copy()

        # frame = imutils.resize(frame, width=2000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        rects = []

        if totalFrames % skip_frame == 0:
            # reset variable
            trackers = []
            labels = []

            start_time_process = time.time()
            numpy_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = offline_deployment(numpy_rgb, verbose=False)
            end_time_process = time.time()

            time_process = end_time_process - start_time_process
            infer_process = "{:.2f}".format(time_process)

            for annot in detections:
                if annot.boxes is not None:
                    for box in annot.boxes:
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        class_name = annot.names[class_id]
                        
                        if class_name in ["Orang", "Motor", "Mobil", "PKL"] :
                            if confidence > 0.4 :
                                # Bounding box (center_x, center_y, width, height)
                                cx, cy, w, h = box.xywh[0].cpu().numpy()
                                center_x = int(cx)
                                center_y = int(cy)
                                width = int(w)
                                height = int(h)
                                left = int(center_x - width / 2)
                                top = int(center_y - height / 2)
                                # cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                                
                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(left, top, left + width, top + height)
                                
                                tracker.start_track(numpy_rgb, rect)
                                trackers.append(tracker)

                                labels.append((class_name))    
                        else :
                            if confidence > confidence_val : 
                                # Bounding box (center_x, center_y, width, height)
                                cx, cy, w, h = box.xywh[0].cpu().numpy()
                                center_x = int(cx)
                                center_y = int(cy)
                                width = int(w)
                                height = int(h)
                                left = int(center_x - width / 2)
                                top = int(center_y - height / 2)
                                # cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                                
                                tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(left, top, left + width, top + height)
                                
                                tracker.start_track(numpy_rgb, rect)
                                trackers.append(tracker)

                                labels.append((class_name))

        else:
            for i in range(len(trackers)):
                tracker = trackers[i]

                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append(((startX, startY, endX, endY), labels[i]))

        objects = ct.update(rects)

        object_id = []

        for (objectID, data) in objects.items():
            to = trackableObjects.get(objectID, None)
            centroid = data[0]
            bbox = data[1]
            label= data[2]
            
            if to is None:
                if pencurian_motor_detection:
                    to = TrackableObject(objectID, centroid, label, bbox, 0, None, False, pencurian_motor_stillness_threshold, movingCallback=movingCallback, roiPencurianMotor=pencurian_motor_roi)
                else:
                    to = TrackableObject(objectID, centroid, label, bbox, 0, None, False, None)
            else:
                ## ==================== PEOPLE COUNTING SECTION ==================== ##
                if counting:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                ## ================================================================= ##
                to.setCentroid(centroid)
                to.frame = frame
                to.bbox = bbox
            ## ==================== OBJECT MENDEKATI OBJECT ==================== ##
            if object_mendekati_object_detection and (label in object_label):
                objectIDs.append(objectID)
            ## ==================== END OF OBJECT MENDEKATI OBJECT ==================== ##
            trackableObjects[objectID] = to

            object_id.append(objectID)
            text = "ID {}".format(objectID)

            cv2.putText(debugFrame, f"{str(objectID)} - {to.name}", (centroid[0] - 10, centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(debugFrame, (centroid[0] - 10, centroid[1] - 10), (centroid[0] + 10, centroid[1] + 10), (0, 255, 0), 3)
            
            
            ## =========================== GERAKAN MENCURIGAKAN DETECTION ============================================= ##
            if gerakan_mencurigakan_detection:
                if not hasattr(to, 'time') or to.time is None:
                                to.time = time.time() 
                if to.name == 'Orang' :
                    if to.isStill : 
                        to.time = time.time()
                    if (time.time() - to.time) > gerakan_mencurigakan_time_limit :
                        imageName2 = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-GM.jpg"
                        to.time = time.time()
                        
                        frameGM = frame.copy()
                        cv2.rectangle(frameGM, (to.bbox[0], to.bbox[1]), (to.bbox[2], to.bbox[3]), (0, 255, 0), 2)

                        _, buffer = cv2.imencode('.jpg', frameGM)

                        # Konversi buffer ke bytes
                        image_bytes = buffer.tobytes()

                        # Encode bytes ke base64
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                        upload_success = helper.uploadFile(image_base64, imageName2)
                        
                        data_result = {
                            "colorRecognition": False,
                            "alpr": False,
                            "objectData": to.name,
                            "objectID": to.objectID,
                            "state": 1,
                            "bbox": to.bbox.tolist(),
                            "context": "orang-ROI-waktu",
                            "class": "person",
                            "daqID": daqID,
                            "camID": camID,
                            "filename": imageName2
                        }
                        if upload_success:
                            messagesQueue.append(data_result)
                        
                        # KIRIM DATA 
                        # helper.send_mqtt(image_base64, data_result, daqID, camID)

                        # print('Gerakan Mencurigakan Sent')
            ## =========================== GERAKAN MENCURIGAKAN DETECTION ============================================= ##
            
            ## =========================== DEMONSTRASI ============================================= ##
            if demonstrasi_detection: 
                if to.name == 'Mobil' :
                    vehicle_counted.append(object_id)
                    
                if to.name == 'Motor':
                    bike_counted.append(object_id)

                if to.name == 'Orang' :
                    people_counted.append(object_id)
                    
                if len(people_counted) > demonstrasi_people_limit or len(bike_counted) > demonstrasi_bike_limit :
                    count_frame_det_demonstrasi += 1
                    
                    if count_frame_det_demonstrasi == 2 :
                        start_time_demonstrasi = time.time()
                else :
                    count_frame_det_demonstrasi = 0
                    start_time_demonstrasi = None
                
                if start_time_demonstrasi is not None:
                    _current_time_demonstrasi_ = time.time()
                    _elapsed_time_demonstrasi = _current_time_demonstrasi_ - start_time_demonstrasi
                    
                    if int(_elapsed_time_demonstrasi) > 10 :
                        _, buffer = cv2.imencode('.jpg', frame)

                        # Konversi buffer ke bytes
                        image_bytes = buffer.tobytes()

                        # Encode bytes ke base64
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        data_result = {
                            "colorRecognition" : False,
                            "alpr" : False,
                            "objectData" : to.name,
                            "objectID" : to.objectID,
                            "state" : to.state,
                            "context" : "kepadatan-kendaraan-orang",
                            "class" : "object",
                            "daqID" : daqID,
                            "camID" : camID,
                            "filename": imageName
                        }
                        messagesQueue.append(data_result)
                        start_time_demonstrasi = time.time()

                        # KIRIM DATA 
                        # helper.send_mqtt(image_base64, data_result, daqID, camID)
                        # print('Demonstrasi Detection Sent')
            ## =========================== END OF DEMONSTRASI ============================================= ##
            ## =========================== PKL ============================================= ##
            if pkl_detection:
                points = np.array(polygon_roi_pkl, np.int32)
                points = points.reshape((-1, 1, 2))
                
                point_to_check = (int(centroid[0]), int(centroid[1]))
                result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
                
                # JIKA ADA DALAM ROI
                if to.name == 'PKL' : 
                    if result > 0:
                        if to.state == 0 :
                            to.state = 1
                            
                            if not hasattr(to, 'time') or to.time is None:
                                to.time = time.time()

                        if to.state == 1 :
                            if (time.time() - to.time) > pkl_time_limit :
                                
                                to.time = time.time()
                                to.status = True
                                
                                _, buffer = cv2.imencode('.jpg', frame)

                                # Konversi buffer ke bytes
                                image_bytes = buffer.tobytes()

                                # Encode bytes ke base64
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
                                data_result = {
                                    "colorRecognition" : False,
                                    "alpr" : False,
                                    "bbox" : to.bbox.tolist(),
                                    "objectData" : to.name,
                                    "objectID" : to.objectID,
                                    "state" : to.state,
                                    "context" : "PKL-ROI-waktu",
                                    "class" : "object",
                                    "daqID" : daqID,
                                    "camID" : camID,
                                    "filename": imageName
                                }
                                
                                messagesQueue.append(data_result)
                                # KIRIM DATA 
                                # helper.send_mqtt(image_base64, data_result, daqID, camID)
                                # print('PKL Detection State 1 Sent')
                                  
                    else :
                        if to.state == 1 :
                            to.state = 0
                            to.time = None
                            if to.status == True:
                                to.status = False
                                    
                                _, buffer = cv2.imencode('.jpg', frame)

                                # Konversi buffer ke bytes
                                image_bytes = buffer.tobytes()

                                # Encode bytes ke base64
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
                                data_result = {
                                    "colorRecognition" : False,
                                    "alpr" : False,
                                    "objectData" : to.name,
                                    "bbox": to.bbox.tolist(),
                                    "state" : to.state,
                                    "objectID" : to.objectID,
                                    "context" : "PKL-ROI-waktu",
                                    "class" : "object",
                                    "daqID" : daqID,
                                    "camID" : camID,
                                    "filename": imageName
                                }
                                
                                messagesQueue.append(data_result)
                                # KIRIM DATA 
                                # helper.send_mqtt(image_base64, data_result, daqID, camID)
                                # print('PKL Detection State 0 Sent')
                            
            ## =========================== END OF PKL ============================================= ##
            
            ## =========================== PENCURIAN MOTOR ============================================= ##
            # if pencurian_motor_detection: 
            #     points = np.array(polygon_roi_pencurian_motor, np.int32)
            #     points = points.reshape((-1, 1, 2))
                
            #     point_to_check = (int(centroid[0]), int(centroid[1]))
            #     result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
                
            #     if to.name == 'Orang' :
            #         if result > 0:
            #             if to.state == 0 :
            #                 # SET WAKTU SAAT OBJEK MASUK KE ROI
            #                 to.state = 1
            #                 to.time = time.time()

            #             if to.state == 1 : 
            #                 if (time.time() - to.time) > pencurian_motor_time_limit :
                                
            #                     to.time = time.time()
                                
            #                     _, buffer = cv2.imencode('.jpg', frame)

            #                     # Konversi buffer ke bytes
            #                     image_bytes = buffer.tobytes()

            #                     # Encode bytes ke base64
            #                     image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
            #                     data_result = {
            #                         "colorRecognition" : False,
            #                         "alpr" : False,
            #                         "objectData" : to.name,
            #                         "objectID" : to.objectID,
            #                         "state" : to.state,
            #                         "bbox": to.bbox.tolist(),
            #                         "context" : "ROI-orang-area-parkir",
            #                         "class" : "person",
            #                         "daqID" : daqID,
            #                         "camID" : camID
            #                     }

            #                     # KIRIM DATA 
            #                     helper.send_mqtt(image_base64, data_result, daqID, camID)
                                
            #                     print()
            #                     print('Pencurian Motor State 1 Sent')
                                       
            #         else :
            #             if to.state == 1 :
            #                 to.state = 0
                            
            #                 to.time = None
                                
            #                 _, buffer = cv2.imencode('.jpg', frame)

            #                 # Konversi buffer ke bytes
            #                 image_bytes = buffer.tobytes()

            #                 # Encode bytes ke base64
            #                 image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
            #                 data_result = {
            #                     "colorRecognition" : True,
            #                     "alpr" : True,
            #                     "objectData" : to.name,
            #                     "bbox": to.bbox.tolist(),
            #                     "state" : to.state,
            #                     "objectID" : to.objectID,
            #                     "context" : "ROI-orang-area-parkir",
            #                     "class" : "person",
            #                     "daqID" : daqID,
            #                     "camID" : camID
            #                 }

            #                 # KIRIM DATA 
            #                 helper.send_mqtt(image_base64, data_result, daqID, camID)
                            
            #                 print()
            #                 print('Pencurian Motor State 0 Sent')
                            
            ## =========================== END OF PENCURIAN MOTOR ============================================= ##           
            ## ================================ PENERTIBAN PARKIR LIAR ====================================== ##
            if parkir_liar_detection :
                points = np.array(polygon_roi, np.int32)
                points = points.reshape((-1, 1, 2))
                
                point_to_check = (int(centroid[0]), int(centroid[1]))
                result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
                
                # JIKA ADA DALAM ROI
                if to.name != 'Orang' and to.name != 'PKL' and to.name != 'No object' and to.name != 'Truk': 
                    if result > 0:
                        if to.state == 0 :
                            to.state = 1
                            
                            if not hasattr(to, 'time') or to.time is None:
                                to.time = time.time()

                        if to.state == 1 :
                            if (time.time() - to.time) > parkir_liar_object_time_limit :
                                to.status = True
                                imageName2 = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-PL.jpg"
                                to.time = time.time()

                                framePL = frame.copy()

                                cv2.rectangle(framePL, (to.bbox[0], to.bbox[1]), (to.bbox[2], to.bbox[3]), (0, 255, 0), 2)
                                
                                _, buffer = cv2.imencode('.jpg', framePL)

                                # Konversi buffer ke bytes
                                image_bytes = buffer.tobytes()

                                # Encode bytes ke base64
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
                                upload_success = helper.uploadFile(image_base64, imageName2)

                                data_result = {
                                    "colorRecognition" : True,
                                    "alpr" : True,
                                    "objectData" : to.name,
                                    "objectID" : to.objectID,
                                    "state" : to.state,
                                    "bbox" : to.bbox.tolist(),
                                    "context" : "kendaraan-ROI-bahujalan-waktu",
                                    "class" : "vehicle",
                                    "daqID" : daqID,
                                    "camID" : camID,
                                    "filename": imageName2
                                }
                                if upload_success:
                                    messagesQueue.append(data_result)
                                # KIRIM DATA 
                                # helper.send_mqtt(image_base64, data_result, daqID, camID)
                                # print(f'PARKIR LIAR 1 State 1 Sent')
                                               
                    if result < 0 :

                        if to.state == 1 :
                            to.state = 0
                            if to.status == True:
                                to.status = False
                                _, buffer = cv2.imencode('.jpg', frame)

                                # Konversi buffer ke bytes
                                image_bytes = buffer.tobytes()

                                # Encode bytes ke base64
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
                                data_result = {
                                    "colorRecognition" : True,
                                    "alpr" : True,
                                    "objectData" : to.name,
                                    "bbox": to.bbox.tolist(),
                                    "state" : to.state,
                                    "objectID" : to.objectID,
                                    "context" : "kendaraan-ROI-bahujalan-waktu",
                                    "class" : "vehicle",
                                    "daqID" : daqID,
                                    "camID" : camID,
                                    "filename": imageName
                                }

                                # KIRIM DATA 
                                # helper.send_mqtt(image_base64, data_result, daqID, camID)
                                # print(f'PARKIR LIAR 1 State 0 Sent')     
                            
            ## =================== END OF PARKIR LIAR ========================== ##
        
        ## ==================== OBJECT MENDEKATI OBJECT ==================== ##
        if object_mendekati_object_detection:
            for i in range(len(objectIDs)):
                for j in range(i + 1, len(objectIDs)):
                    objectA = trackableObjects.get(objectIDs[i], None)
                    objectB = trackableObjects.get(objectIDs[j], None)
                    centroidA = objectA.centroids[-1]
                    centroidB = objectB.centroids[-1]
                    distance = np.linalg.norm(np.array(centroidA) - np.array(centroidB))
                    # helper.writelog(sensor_id, f"{objectA.name}({objectA.objectID}),{objectB.name}({objectB.objectID}): {distance:.1f}")
                    # print(f"{objectA.name}({objectA.objectID}),{objectB.name}({objectB.objectID}): {distance:.1f}")
                    # cv2.putText(debugFrame, f"{distance:.1f}", (centroidA[0] - 10, centroidA[1] - (20+(10*i))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(debugFrame, f"{distance:.1f}", (centroidB[0] - 10, centroidB[1] - (20+(10*j))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # if not hasattr(to, 'timeMendekati') or to.timeMendekati is None:
                    #     to.timeMendekati = time.time()
                    
                    con1 = "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang"
                    con2 = "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran"
                    if distance <= distance_threshold and (totalFrames-lastSend > 30):
                        ## OBJECT A
                        _, buffer = cv2.imencode('.jpg', frame)

                        # Konversi buffer ke bytes
                        image_bytes = buffer.tobytes()

                        # Encode bytes ke base64
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                        data_result = {
                            "colorRecognition" : True if objectA.name == motor_label else False,
                            "alpr" : True if objectA.name == motor_label else False,
                            "objectID" : objectA.objectID,
                            "objectData" : objectA.name,
                            "bbox": objectA.bbox.tolist(),
                            "state" : 1,
                            "context" : "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang",
                            "class" : "vehicle" if (objectA.name == motor_label) else "person",
                            "daqID" : daqID,
                            "camID" : camID,
                            "filename": imageName
                        }
                        # print(objectA.name, objectB.name)
                        # print("motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang")
                        # cv2.putText(debugFrame, "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang", (centroidA[0] - 10, centroidA[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # cv2.putText(debugFrame, "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang", (centroidB[0] - 10, centroidB[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        messagesQueue.append(data_result)
                        # KIRIM DATA 
                        # helper.send_mqtt(image_base64, data_result, daqID, camID)

                        if pencurian_motor_detection:
                            points = np.array(pencurian_motor_roi, np.int32)
                            points = points.reshape((-1, 1, 2))
                            
                            point_to_checkA = (int(centroidA[0]), int(centroidA[1]))
                            resultA = cv2.pointPolygonTest(points, point_to_checkA, measureDist=False)

                            point_to_checkA = (int(centroidB[0]), int(centroidB[1]))
                            resultB = cv2.pointPolygonTest(points, point_to_checkA, measureDist=False)

                            if resultA > 0 and resultB > 0:
                                data_result = {
                                    "colorRecognition" : True if objectA.name == motor_label else False,
                                    "alpr" : True if objectA.name == motor_label else False,
                                    "objectID" : objectA.objectID,
                                    "objectData" : objectA.name,
                                    "bbox": objectA.bbox.tolist(),
                                    "state" : 1,
                                    "context" : "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran",
                                    "class" : "vehicle" if (objectA.name == motor_label) else "person",
                                    "daqID" : daqID,
                                    "camID" : camID,
                                    "filename": imageName
                                }
                                # print(objectA.name, objectB.name)
                                # print("motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran")
                                # cv2.putText(debugFrame, "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran", (centroidA[0] - 10, centroidA[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                # cv2.putText(debugFrame, "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran", (centroidB[0] - 10, centroidB[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                messagesQueue.append(data_result)
                                # KIRIM DATA 
                                # helper.send_mqtt(image_base64, data_result, daqID, camID)
                        
                        ## OBJECT B
                        # _, buffer = cv2.imencode('.jpg', frame)

                        # # Konversi buffer ke bytes
                        # image_bytes = buffer.tobytes()

                        # # Encode bytes ke base64
                        # image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # data_result = {
                        #     "colorRecognition" : True if objectB.name == motor_label else False,
                        #     "alpr" : True if objectB.name == motor_label else False,
                        #     "objectID" : objectB.objectID,
                        #     "objectData" : objectB.name,
                        #     "bbox": objectB.bbox.tolist(),
                        #     "state" : 1,
                        #     "context" : "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang",
                        #     "class" : "vehicle" if (objectB.name == motor_label) else "person",
                        #     "daqID" : daqID,
                        #     "camID" : camID,
                        #     "filename": imageName
                        # }
                        # print("motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang")
                        # messagesQueue.append(data_result)
                        # # KIRIM DATA 
                        # # helper.send_mqtt(image_base64, data_result, daqID, camID)

                        # if pencurian_motor_detection:
                        #     points = np.array(pencurian_motor_roi, np.int32)
                        #     points = points.reshape((-1, 1, 2))
                            
                        #     point_to_checkA = (int(centroidA[0]), int(centroidA[1]))
                        #     resultA = cv2.pointPolygonTest(points, point_to_checkA, measureDist=False)

                        #     point_to_checkA = (int(centroidB[0]), int(centroidB[1]))
                        #     resultB = cv2.pointPolygonTest(points, point_to_checkA, measureDist=False)

                        #     if resultA > 0 and resultB > 0:
                        #         data_result = {
                        #             "colorRecognition" : True if objectB.name == motor_label else False,
                        #             "alpr" : True if objectB.name == motor_label else False,
                        #             "objectID" : objectB.objectID,
                        #             "objectData" : objectB.name,
                        #             "bbox": objectB.bbox.tolist(),
                        #             "state" : 1,
                        #             "context" : "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran",
                        #             "class" : "vehicle" if (objectB.name == motor_label) else "person",
                        #             "daqID" : daqID,
                        #             "camID" : camID
                        #         }
                        #         print("motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran")
                        #         messagesQueue.append(data_result)
                        #         # KIRIM DATA 
                        #         # helper.send_mqtt(image_base64, data_result, daqID, camID)
                            
                        
                        lastSend = totalFrames
                        
        ## ==================== END OF MOTOR MENDEKATI MOTOR ==================== ##
        ## ================================================================= ##        
        
        if len(messagesQueue) > 0:
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            upload_success = helper.uploadFile(image_base64, imageName)

            if upload_success:
                for message in messagesQueue:
                    helper.send_mqtt(message)
                    print(f"Send {message['context']}")
            
            messagesQueue = []


        points_roi = np.array(polygon_roi_pkl, np.int32)
        points_roi = points_roi.reshape((-1, 1, 2))
        cv2.polylines(debugFrame, [points_roi], isClosed=True, color=(0, 255, 0), thickness=5)
        
        _, buffer = cv2.imencode('.jpg', debugFrame)
        frame_b64 = base64.b64encode(buffer)
        footage_socket.send(frame_b64)
        
        # out.write(cv2.resize(debugFrame, (frame_width,frame_height)))

        # ROI PKL
        points_roi = np.array(polygon_roi_pkl, np.int32)
        points_roi = points_roi.reshape((-1, 1, 2))
        cv2.polylines(frame, [points_roi], isClosed=True, color=(0, 255, 0), thickness=2)

        # ROI Parkir Liar
        points_roi = np.array(polygon_roi, np.int32)
        points_roi = points_roi.reshape((-1, 1, 2))
        cv2.polylines(frame, [points_roi], isClosed=True, color=(255, 0, 0), thickness=2)

        # ROI Pencurian Motor
        points_roi = np.array(polygon_roi_pencurian_motor, np.int32)
        points_roi = points_roi.reshape((-1, 1, 2))
        cv2.polylines(frame, [points_roi], isClosed=True, color=(0, 0, 255), thickness=2)
        
        cv2.imwrite('ENV.jpg', frame)
        # newf = cv2.resize(frame, (1280, 720))
        # cv2.imshow("TEST", newf)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
        totalFrames += 1
        fps.update()

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed())) if verbose else None
    logger.info("Approx. FPS: {:.2f}".format(fps.fps())) if verbose else None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--sensor_id", help="")
    args = parser.parse_args()
    
    sensor_id = args.sensor_id
    config = helper.loadConfig(sensor_id)

    offline_deployment = YOLO(config["config_local"]["deployment"])

    main(
        config["config_api"]["rtsp_url"], 
        offline_deployment, 
        sensor_id
    )