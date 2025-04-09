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
    
    # ## ================= Config Objek Tertinggal Object ================= ##
    # objek_tertinggal_detection = helper.loadConfig(sensor_id)["config_barang_tertinggal"]["gerakan_mencurigakan_detection"]
    # objek_tertinggal_time_limit = helper.loadConfig(sensor_id)["config_barang_tertinggal"]["object_time_limit"]
    
    ## ================= Config PKL Object ================= ##
    pkl_detection = helper.loadConfig(sensor_id)["config_pkl"]["pkl_detection"]
    pkl_time_limit = helper.loadConfig(sensor_id)["config_pkl"]["object_time_limit"]
    polygon_roi_pkl = helper.loadConfig(sensor_id)["config_pkl"]["polygon_roi_pkl"]
    
    ## ================= Config API =================
    zmq_address = helper.loadConfig(sensor_id)["config_api"]["zmq_address"]
    serviceStandalone = helper.loadConfig(sensor_id)["config_api"]["urlService"]

    logger.info("Starting the video..") if verbose else None

    vs = cv2.VideoCapture(video_path)

    W = H = None

    ct = CentroidTracker(maxDisappeared=50, maxDistance=80)

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
        objectIDs = []
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
    
    while True:

        hasFrame, frame = vs.read()
        
        if frame is not None and frame.size > 0:
            frame = frame
        else:
            print("Warning: Frame is empty or invalid.")
        
        if not hasFrame or frame is None:
            frame = np.zeros((1020, 480, 3), dtype=np.uint8)
            vs.release()
            vs = cv2.VideoCapture(video_path)
            logger.info('CAM ERROR') if verbose else None

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
            detections = offline_deployment.infer(numpy_rgb)
            end_time_process = time.time()

            time_process = end_time_process - start_time_process
            infer_process = "{:.2f}".format(time_process)

            for annot in detections.annotations:
                for lab in annot.labels:
                    scores = lab.probability
                    classId = lab.name
                    confidence = scores

                    if confidence > confidence_val:
                        ## Code deployment bounding box
                        
                        x = annot.shape.x
                        y = annot.shape.y
                        w = annot.shape.width
                        h = annot.shape.height

                        center_x = int(annot.shape.x * W)
                        center_y = int(annot.shape.y * H)
                        width = int(annot.shape.width * W)
                        height = int(annot.shape.height * H)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        
                        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                        
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(int(annot.shape.x), int(annot.shape.y), int(annot.shape.x + annot.shape.width), int(annot.shape.y + annot.shape.height))
                        
                        tracker.start_track(numpy_rgb, rect)
                        trackers.append(tracker)

                        labels.append((lab.name))

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
                to = TrackableObject(objectID, centroid, label, bbox, 0, None)
                # to.setPlate(plates)
            else:
                ## ==================== PEOPLE COUNTING SECTION ==================== ##
                if counting:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                ## ================================================================= ##
                to.centroids.append(centroid)
                to.bbox = bbox
            ## ==================== OBJECT MENDEKATI OBJECT ==================== ##
            if object_mendekati_object_detection and (label in object_label):
                objectIDs.append(objectID)
            ## ==================== END OF OBJECT MENDEKATI OBJECT ==================== ##
            trackableObjects[objectID] = to

            object_id.append(objectID)
            text = "ID {}".format(objectID)

            cv2.rectangle(frame, (centroid[0] - 10, centroid[1] - 10), (centroid[0] + 10, centroid[1] + 10), (0, 255, 0), 3)
            
            ## =========================== DEMONSTRASI ============================================= ##
            if demonstrasi_detection: 
                if to.name == 'Mobil' :
                    vehicle_counted.append(classId)
                    
                if to.name == 'Motor':
                    bike_counted.append(classId)

                if to.name == 'Orang' :
                    people_counted.append(classId)
                    
                if len(people_counted) > demonstrasi_people_limit and len(bike_counted) > demonstrasi_bike_limit :
                    count_frame_det_demonstrasi += 1
                    
                    if count_frame_det_demonstrasi == 2 :
                        start_time_demonstrasi = time.time()
                else :
                    count_frame_det_demonstrasi = 0
                    start_time_demonstrasi = None
                
                if start_time_demonstrasi is not None:
                    _current_time_demonstrasi_ = time.time()
                    _elapsed_time_demonstrasi = _current_time_demonstrasi_ - start_time_demonstrasi
                    
                    if int(_elapsed_time_demonstrasi) > 60 :
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
                            "image" : image_base64,
                            "context" : "kepadatan-kendaraan-orang",
                            "class" : "object",
                            "daqID" : daqID,
                            "camID" : camID
                        }
                        
                        print()
                        print('DEMO')
                        # print(data_result)

                        url = serviceStandalone
                        headers = {'Content-Type': 'application/json'}
                        
                        response = requests.post(url, headers=headers, json=data_result)  
                        if response.status_code == 200:
                            try:
                                print("INI HASIL RESPONSE API", response.json())
                                count_frame_det_demonstrasi = 0
                                start_time_demonstrasi = None
                            except ValueError:
                                print("Response bukan format JSON:", response.text)
                        else:
                            print("Error pada request:", response.status_code, response.text) 
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
                                    
                            _current_time_obj_pkl = time.time()
                            _elapsed_time_obj_pkl = _current_time_obj_pkl - to.time

                        if to.state == 1 :
                            if _elapsed_time_obj_pkl > pkl_time_limit :
                                
                                to.time = time.time()
                                
                                # cropped_image = frame[to.bbox[1]:to.bbox[3], to.bbox[0]:to.bbox[2]]
                                
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
                                    "image" : image_base64,
                                    "context" : "PKL-ROI-waktu",
                                    "class" : "object",
                                    "daqID" : daqID,
                                    "camID" : camID
                                }
                                
                                print()
                                print('PKL')
                                # print(data_result)

                                url = serviceStandalone
                                headers = {'Content-Type': 'application/json'}
                                
                                response = requests.post(url, headers=headers, json=data_result)  
                                if response.status_code == 200:
                                    try:
                                        print("INI HASIL RESPONSE API", response.json())
                                        to.time = time.time()
                                    except ValueError:
                                        print("Response bukan format JSON:", response.text)
                                else:
                                    print("Error pada request:", response.status_code, response.text)     
                    else :
                        if to.state == 1 :
                            to.state = 0
                            
                            to.time = None
                            
                            _elapsed_time_obj_pkl = None
                            
                            # cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                
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
                                "image" : image_base64,
                                "state" : to.state,
                                "objectID" : to.objectID,
                                "context" : "PKL-ROI-waktu",
                                "class" : "object",
                                "daqID" : daqID,
                                "camID" : camID
                            }
                            
                            url = serviceStandalone
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, headers=headers, json=data_result)  
                            if response.status_code == 200:
                                try:
                                    print("INI HASIL RESPONSE API", response.json())
                                except ValueError:
                                    print("Response bukan format JSON:", response.text)
                            else:
                                print("Error pada request:", response.status_code, response.text)
            ## =========================== END OF PKL ============================================= ##
            
            ## =========================== GERAKAN MENCURIGAKAN ============================================= ##
            if pencurian_motor_detection: 
                points = np.array(polygon_roi_pencurian_motor, np.int32)
                points = points.reshape((-1, 1, 2))
                
                point_to_check = (int(centroid[0]), int(centroid[1]))
                result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
                
                if to.name == 'Orang' :
                    if result > 0:
                        if to.state == 0 :
                            # SET WAKTU SAAT OBJEK MASUK KE ROI
                            to.state = 1
                            to.time = time.time()

                            if to.state == 1 : 
                                _current_time_obj_pencurian_motor = time.time()
                                _elapsed_time_obj_pencurian_motor = _current_time_obj_pencurian_motor - to.time
                                
                                if _elapsed_time_obj_pencurian_motor > pencurian_motor_time_limit :
                                    
                                    to.time = time.time()
                                    
                                    # cropped_image = frame[to.bbox[1]:to.bbox[3], to.bbox[0]:to.bbox[2]]
                                    
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
                                        "bbox": to.bbox.tolist(),
                                        "image" : image_base64,
                                        "context" : "ROI-orang-area-parkir",
                                        "class" : "person",
                                        "daqID" : daqID,
                                        "camID" : camID
                                    }
                                    
                                    print()
                                    print('GERAKAN MENCURIGAKAN ORANG')
                                    # print(data_result)

                                    url = serviceStandalone
                                    headers = {'Content-Type': 'application/json'}
                                    
                                    response = requests.post(url, headers=headers, json=data_result)  
                                    if response.status_code == 200:
                                        try:
                                            print("INI HASIL RESPONSE API", response.json())
                                        except ValueError:
                                            print("Response bukan format JSON:", response.text)
                                    else:
                                        print("Error pada request:", response.status_code, response.text)     
                    else :
                        if to.state == 1 :
                            to.state = 0
                            
                            to.time = None
                            
                            _elapsed_time_obj_gerakan_mencurigakan = None
                            
                            # cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                
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
                                "image" : image_base64,
                                "state" : to.state,
                                "objectID" : to.objectID,
                                "context" : "ROI-orang-area-parkir",
                                "class" : "person",
                                "daqID" : daqID,
                                "camID" : camID
                            }
                            
                            print()
                            print('GERAKAN MENCURIGAKAN !ORANG')
                            # print(data_result)

                            url = serviceStandalone
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, headers=headers, json=data_result)  
                            if response.status_code == 200:
                                try:
                                    print("INI HASIL RESPONSE API", response.json())
                                except ValueError:
                                    print("Response bukan format JSON:", response.text)
                            else:
                                print("Error pada request:", response.status_code, response.text)
            ## =========================== END OF GERAKAN MENCURIGAKAN ============================================= ##           
            ## ================================ PENERTIBAN PARKIR LIAR ====================================== ##
            if parkir_liar_detection :
                points = np.array(polygon_roi, np.int32)
                points = points.reshape((-1, 1, 2))
                
                point_to_check = (int(centroid[0]), int(centroid[1]))
                result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
                
                # JIKA ADA DALAM ROI
                if to.name != 'Orang' and to.name != 'PKL' and to.name != 'No object': 
                    if result > 0:
                        if to.state == 0 :
                            to.state = 1
                            
                            if not hasattr(to, 'time') or to.time is None:
                                to.time = time.time()

                            _current_time_obj_ = time.time()
                            _elapsed_time_obj_parkir_liar = _current_time_obj_ - to.time

                            print("Timer : ", _elapsed_time_obj_parkir_liar)
                            print("Object State : ", to.state)

                        if to.state == 1 :
                            if (time.time() - to.time) > parkir_liar_object_time_limit :
                                # print(f"TIME PARKIR LIAR 1: {(time.time() - to.time)}")
                                to.time = time.time()
                                
                                # cropped_image = frame[to.bbox[1]:to.bbox[3], to.bbox[0]:to.bbox[2]]
                                
                                _, buffer = cv2.imencode('.jpg', frame)

                                # Konversi buffer ke bytes
                                image_bytes = buffer.tobytes()

                                # Encode bytes ke base64
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                
                                data_result = {
                                    "colorRecognition" : True,
                                    "alpr" : True,
                                    "objectData" : to.name,
                                    "objectID" : to.objectID,
                                    "state" : to.state,
                                    "bbox" : to.bbox.tolist(),
                                    "image" : image_base64,
                                    "context" : "kendaraan-ROI-bahujalan-waktu",
                                    "class" : "vehicle",
                                    "daqID" : daqID,
                                    "camID" : camID
                                }
                                # logger.info(f"Elapse State 1:{data_result}")
                                
                                print()
                                print('PARKIR LIAR 1')
                                # print(data_result)

                                url = serviceStandalone
                                headers = {'Content-Type': 'application/json'}
                                
                                response = requests.post(url, headers=headers, json=data_result)  
                                if response.status_code == 200:
                                    try:
                                        print("INI HASIL RESPONSE API", response.json())
                                        to.time = time.time()
                                    except ValueError:
                                        print("Response bukan format JSON:", response.text)
                                else:
                                    print("Error pada request:", response.status_code, response.text)     
                    else :
                        if to.state == 1 :
                            to.state = 0
                            
                            # to.time = None
                            
                            # _elapsed_time_obj_parkir_liar = None
                            
                            # cropped_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                
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
                                "image" : image_base64,
                                "state" : to.state,
                                "objectID" : to.objectID,
                                "context" : "kendaraan-ROI-bahujalan-waktu",
                                "class" : "vehicle",
                                "daqID" : daqID,
                                "camID" : camID
                            }
                            
                            # logger.info(f"TO State 1: {data_result}")
                            print()
                            print('PARKIR LIAR 2')
                            # print(data_result)

                            url = serviceStandalone
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, headers=headers, json=data_result)  
                            if response.status_code == 200:
                                try:
                                    print("INI HASIL RESPONSE API", response.json())
                                except ValueError:
                                    print("Response bukan format JSON:", response.text)
                            else:
                                print("Error pada request:", response.status_code, response.text)
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

                    objectA.time = time.time()
                    objectB.time = time.time()
                                    
                    _current_time_obj_a = time.time()
                    _elapsed_time_obj_a = _current_time_obj_a - objectA.time

                    _current_time_obj_b = time.time()
                    _elapsed_time_obj_b = _current_time_obj_b - objectB.time


                    if (_current_time_obj_a % 10 == 0) or (_current_time_obj_b % 10 == 0) :
                        if distance > distance_threshold and (totalFrames-lastSend) > 30:
                            ## OBJECT A
                            valA = objectA.bbox
                            # cropped_image = frame[valA[1]:valA[3], valA[0]:valA[2]]
                            
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
                                "image" : image_base64,
                                "state" : 1,
                                "context" : "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang",
                                "class" : "vehicle" if (objectA.name == motor_label) else "person",
                                "daqID" : daqID,
                                "camID" : camID
                            }
                            
                            print()
                            print('OBJEK MENDEKATI OBJEK A')
                            # print(data_result)

                            url = serviceStandalone
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, headers=headers, json=data_result)  
                            if response.status_code == 200:
                                try:
                                    print("INI HASIL RESPONSE API", response.json())
                                except ValueError:
                                    print("Response bukan format JSON:", response.text)
                            else:
                                print("Error pada request:", response.status_code, response.text)
                            
                            ## OBJECT B
                            valB = objectB.bbox
                            # cropped_image = frame[valB[1]:valB[3], valB[0]:valB[2]]
                            
                            _, buffer = cv2.imencode('.jpg', frame)

                            # Konversi buffer ke bytes
                            image_bytes = buffer.tobytes()

                            # Encode bytes ke base64
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            data_result = {
                                "colorRecognition" : True if objectB.name == motor_label else False,
                                "alpr" : True if objectB.name == motor_label else False,
                                "objectID" : objectB.objectID,
                                "objectData" : objectB.name,
                                "bbox": objectB.bbox.tolist(),
                                "image" : image_base64,
                                "state" : 1,
                                "context" : "motor-mendekati-motor" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang",
                                "class" : "vehicle" if (objectB.name == motor_label) else "person",
                                "daqID" : daqID,
                                "camID" : camID
                            }
                            
                            print()
                            print('OBJEK MENDEKATI OBJEK B')
                            # print(data_result)

                            url = serviceStandalone
                            headers = {'Content-Type': 'application/json'}
                            
                            response = requests.post(url, headers=headers, json=data_result)  
                            if response.status_code == 200:
                                try:
                                    print("INI HASIL RESPONSE API", response.json())
                                except ValueError:
                                    print("Response bukan format JSON:", response.text)
                            else:
                                print("Error pada request:", response.status_code, response.text)
                            lastSend = totalFrames
        ## ==================== END OF MOTOR MENDEKATI MOTOR ==================== ##
        ## ================================================================= ##        
        
        points_roi = np.array(polygon_roi, np.int32)
        points_roi = points_roi.reshape((-1, 1, 2))
        cv2.polylines(frame, [points_roi], isClosed=True, color=(0, 255, 0), thickness=5)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer)
        footage_socket.send(frame_b64)
        
        cv2.imwrite('test.jpg', frame)

        # cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        
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

    offline_deployment = Deployment.from_folder(
        config['config_local']['main_path'] + config['config_local']['deployment']
    )
    offline_deployment.load_inference_models(device="CPU")

    main(
        config["config_api"]["rtsp_url"], 
        offline_deployment, 
        sensor_id
    )