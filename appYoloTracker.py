import argparse
import cv2
import logging
import numpy as np
from imutils.video import FPS
from ultralytics import YOLO
import helper
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import threading
import time 
import os
from tracker.trackableobject import TrackableObject
import base64
import zmq
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_executor = ThreadPoolExecutor(max_workers=4)
metric_executor = ThreadPoolExecutor(max_workers=4)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

track_history = defaultdict(lambda: {"centroids": [], "class_name": None, "bbox": []})
trackableObjects = {}
detected_ids = set()

global start_time_demonstrasi
global start_time_omo
global omoDetected
global totalFrames
global omoLastSend
global verbose
global daqID
global camID


start_time_demonstrasi = None
start_time_omo = None
omoDetected = []
totalFrames = 0
omoLastSend = 0
verbose = False
daqID = ""
camID = ""

def send_data(image_base64, image_name, message, topic="analytic-message"):
    """Function to upload the image asynchronously."""
    try:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        upload_success = helper.uploadFile(image_base64, image_name)
        if upload_success:
            if "context" in message:
                message["filename"] = image_name
            else:
                message["file"] = image_name
            
            helper.send_mqtt(message, topic=topic)
            if "context" in message:
                logger.info(f"[{current_datetime}] Send {message['context']}") if verbose else None
            else:
                logger.info(f"[{current_datetime}] Send {message['objects'][0]['id']}") if verbose else None
        else:
            logger.error(f"Failed to upload {image_name}")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")

def save_metric(sensor_id):
    try:
        helper.writelog(sensor_id,f"Ping: {helper.get_ping('10.11.5.160')}")
        helper.writelog(sensor_id,f"CPU: {helper.get_cpu_usage()}")
        helper.writelog(sensor_id,f"Memory: {helper.get_memory_usage()}")
    except Exception as e:
        logger.error(f"Error in retrieving metrics: {e}")

def movingCallback(self, centroid):
    global daqID
    global camID

    points = np.array(self.roiPencurianMotor, np.int32)
    points = points.reshape((-1, 1, 2))
    
    point_to_check = (int(centroid[0]), int(centroid[1]))
    result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)

    _, buffer = cv2.imencode('.jpg', self.frame)
    image_bytes = buffer.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    imageName = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-PM.jpg"
    
    if result > 0:
        data_result = {
            "colorRecognition" : False,
            "alpr" : False,
            "objectData" : self.name,
            "objectID" : int(self.objectID),
            "state" : 1,
            "bbox": [int(x) for x in self.bbox[-1]],
            "context" : f"{self.name.lower()}-bergerak-ROI-parkiran",
            "class" : "person" if self.name == "Orang" else "vehicle",
            "daqID" : daqID,
            "camID" : camID,
            "filename": imageName
        }

        # KIRIM DATA
        data_executor.submit(send_data, image_base64, imageName, data_result)

def process_detections(frame, config):
    ## Load global variables
    global start_time_demonstrasi
    global start_time_omo
    global omoDetected
    global totalFrames
    global omoLastSend
    global daqID
    global camID

    ## LOAD CONFIG ##
    ## ================= Config Devices =================
    confidence_per_class = config["label_confidence"]
    
    ## ================= Config Local =================
    verbose = config["config_local"]["verbose"]
    confidence_val = config["config_local"]["confidence_val"]
    
    ## ================= Config Parkir Liar =================
    parkir_liar_detection = config["config_parkir_liar"]["parkir_liar_detection"]
    polygon_roi = config["config_parkir_liar"]["polygon_roi"]
    parkir_liar_object_time_limit = config["config_parkir_liar"]["object_time_limit"]

    ## ================= Config Object Mendekati Object =================
    object_mendekati_object_detection = config["config_object_mendekati_object"]["object_mendekati_object_detection"]
    omo_distance_threshold = config["config_object_mendekati_object"]["distance_threshold"]
    omo_delay = config["config_object_mendekati_object"]["delay"]
    object_label = config["config_object_mendekati_object"]["object_label"]
    motor_label = config["config_object_mendekati_object"]["motor_label"]
    orang_label = config["config_object_mendekati_object"]["orang_label"]
    
    ## ================= Config Demonstrasi Object ================= ##
    demonstrasi_detection = config["config_demonstrasi"]["demonstrasi_detection"]
    demonstrasi_vehicle_limit = config["config_demonstrasi"]["demonstrasi_vehicle_limit"]
    demonstrasi_bike_limit = config["config_demonstrasi"]["demonstrasi_bike_limit"]
    demonstrasi_people_limit = config["config_demonstrasi"]["demonstrasi_people_limit"]
    demonstrasi_roi = config["config_demonstrasi"]["polygon_roi_demonstrasi"]

    ## ================= Config Gerakan Mencurigakan ================= ##
    gerakan_mencurigakan_detection = config["config_gerakan_mencurigakan"]["gerakan_mencurigakan_detection"]
    gerakan_mencurigakan_time_limit = config["config_gerakan_mencurigakan"]["object_time_limit"]
    gerakan_mencurigakan_roi = config["config_gerakan_mencurigakan"]["polygon_roi_gerakan_mencurigakan"]
    
    ## ================= Config PKL Object ================= ##
    pkl_detection = config["config_pkl"]["pkl_detection"]
    pkl_time_limit = config["config_pkl"]["object_time_limit"]
    polygon_roi_pkl = config["config_pkl"]["polygon_roi_pkl"]

    ## ================= Config Pencurian Motor ================= ##
    pencurian_motor_detection = config["config_pencurian_motor"]["pencurian_motor_detection"]
    pencurian_motor_stillness_threshold = config["config_pencurian_motor"]["stillness_threshold"]
    pencurian_motor_roi = config["config_pencurian_motor"]["polygon_roi_pencurian_motor"]

    ## ==================== OBJECT MENDEKATI OBJECT SECTION ==================== ##
    if object_mendekati_object_detection:
        objectIDs = []

    messagesQueue = []
    messageCounter = 0

    ## INITIALIZATION VAR FOR DETECTION ##
    vehicle_counted = []
    bike_counted = []
    people_counted = []

    imageName = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}.jpg"

    ## ========= LOAD MODEL AND TRACKER =============== ##
    detections = offline_deployment.track(frame, persist=True, tracker="tracker.yaml", verbose=False)

    ## ========== DETECTION PROCESS ================ ##
    for annot in detections:
        boxes = annot.boxes.xyxy.cpu().numpy().astype(int)  # Ambil semua bounding box
        ids = annot.boxes.id.cpu().numpy().astype(int) if annot.boxes.id is not None else np.arange(len(boxes))  # Track ID
        confidences = annot.boxes.conf.cpu().numpy()
        class_ids = annot.boxes.cls.cpu().numpy().astype(int)
        for box, track_id, confidence, class_id in zip(boxes, ids, confidences, class_ids):
            class_name = annot.names[class_id]
            class_confidence_threshold = confidence_per_class.get(class_name.lower(), confidence_val)
            if confidence > class_confidence_threshold:
                xmin, ymin, xmax, ymax = box

                # Update tracking history
                if track_id not in track_history:
                    track_history[track_id] = {
                        "centroids": [],
                        "class_name": class_name,
                        "bbox": [],
                        "last_seen": 0
                    }

                track_history[track_id]["centroids"].append(((xmin + xmax) / 2, (ymin + ymax) / 2))
                track_history[track_id]["class_name"] = class_name
                track_history[track_id]["bbox"].append((xmin, ymin, xmax, ymax))
                track_history[track_id]["last_seen"] = 0

                if len(track_history[track_id]["centroids"]) > 30:
                    track_history[track_id]["centroids"].pop(0)

    for track_id in list(track_history.keys()):
        if track_id not in detected_ids:
            track_history[track_id]["last_seen"] += 1
            if track_history[track_id]["last_seen"] > 30:
                del track_history[track_id]

    for track_id, data in track_history.items():
        centroids = data["centroids"]
        class_name = data["class_name"]
        bbox = data["bbox"]

        to = trackableObjects.get(track_id, None)

        if to is None :
            if pencurian_motor_detection:
                to = TrackableObject(track_id, centroids, class_name, bbox, 0, None, False, pencurian_motor_stillness_threshold, movingCallback=movingCallback, roiPencurianMotor=pencurian_motor_roi)
            else:
                to = TrackableObject(track_id, centroids, class_name, bbox, 0, None, False, None)
        else :
            if len(centroids) > 1:
                to.setCentroid(centroids[-1])
            elif len(centroids) == 1:
                to.setCentroid(centroids[0])
            to.frame = frame
            to.bbox = bbox 
        
        if object_mendekati_object_detection and (class_name in object_label):
                objectIDs.append(track_id)
        trackableObjects[track_id] = to

        if centroids and bbox:
            x1, y1, x2, y2 = to.bbox[-1]  # Ambil koordinat bounding box
            cx, cy = centroids[-1]
            cv2.putText(frame, str(to.name), (int(cx) - 10, int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # if to.time is not None :
            #     cv2.putText(frame, str(time.time() - to.time), (int(cx) - 30, int(cy) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            

        ## ==================== OBJECT MENDEKATI OBJECT ==================== ##
        if object_mendekati_object_detection:
            if start_time_omo is None:
                start_time_omo = time.time()
            for i in range(len(objectIDs)):
                for j in range(i + 1, len(objectIDs)):
                    objectA = trackableObjects.get(objectIDs[i], None)
                    objectB = trackableObjects.get(objectIDs[j], None)
                    centroidA = objectA.centroids[-1]
                    centroidB = objectB.centroids[-1]
                    distance = np.linalg.norm(np.array(centroidA) - np.array(centroidB))

                    # if distance <= omo_distance_threshold and (omoLastSend-totalFrames) > omo_delay:
                    if distance <= omo_distance_threshold:
                        ## OBJECT A
                        if f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}" not in omoDetected:
                            data_result = {
                                "colorRecognition" : True if objectA.name == motor_label else False,
                                "alpr" : True if objectA.name == motor_label else False,
                                "objectID" : int(objectA.objectID),
                                "objectData" : objectA.name,
                                "bbox": [int(x) for x in objectA.bbox[-1]],
                                "state" : 1,
                                "context" : f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}",
                                "class" : "vehicle" if (objectA.name == motor_label) else "person",
                                "daqID" : daqID,
                                "camID" : camID,
                                "filename": imageName
                            }
                            messagesQueue.append(data_result)
                            omoDetected.append(f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}")
                            omoLastSend = totalFrames

                        ## OBJECT B
                        if f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}" not in omoDetected:
                            data_result = {
                                "colorRecognition" : True if objectB.name == motor_label else False,
                                "alpr" : True if objectB.name == motor_label else False,
                                "objectID" : int(objectB.objectID),
                                "objectData" : objectB.name,
                                "bbox": [int(x) for x in objectB.bbox[-1]],
                                "state" : 1,
                                "context" : f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}",
                                "class" : "vehicle" if (objectB.name == motor_label) else "person",
                                "daqID" : daqID,
                                "camID" : camID,
                                "filename": imageName
                            }
                            messagesQueue.append(data_result)
                            omoDetected.append(f"{objectA.name.lower()}-mendekati-{objectB.name.lower()}")
                            omoLastSend = totalFrames

                        if pencurian_motor_detection:
                            points = np.array(pencurian_motor_roi, np.int32)
                            points = points.reshape((-1, 1, 2))
                            
                            point_to_checkA = (int(centroidA[0]), int(centroidA[1]))
                            resultA = cv2.pointPolygonTest(points, point_to_checkA, measureDist=False)

                            point_to_checkB = (int(centroidB[0]), int(centroidB[1]))
                            resultB = cv2.pointPolygonTest(points, point_to_checkB, measureDist=False)

                            if resultA > 0 and resultB > 0:
                                if (objectA.name == motor_label) and (objectB.name == motor_label) and "motor-mendekati-motor-ROI-parkiran" not in omoDetected:
                                    data_result = {
                                        "colorRecognition" : True if objectA.name == motor_label else False,
                                        "alpr" : True if objectA.name == motor_label else False,
                                        "objectID" : int(objectA.objectID),
                                        "objectData" : objectA.name,
                                        "bbox": [int(x) for x in objectA.bbox[-1]],
                                        "state" : 1,
                                        "context" : "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran",
                                        "class" : "vehicle" if (objectA.name == motor_label) else "person",
                                        "daqID" : daqID,
                                        "camID" : camID,
                                        "filename": imageName
                                    }
                                    messagesQueue.append(data_result)
                                    omoDetected.append("motor-mendekati-motor-ROI-parkiran")
                                    omoLastSend = totalFrames
                                if not ((objectA.name == motor_label) and (objectB.name == motor_label)) and "motor-mendekati-orang-ROI-parkiran" not in omoDetected:
                                    data_result = {
                                        "colorRecognition" : True if objectA.name == motor_label else False,
                                        "alpr" : True if objectA.name == motor_label else False,
                                        "objectID" : int(objectA.objectID),
                                        "objectData" : objectA.name,
                                        "bbox": [int(x) for x in objectA.bbox[-1]],
                                        "state" : 1,
                                        "context" : "motor-mendekati-motor-ROI-parkiran" if (objectA.name == motor_label) and (objectB.name == motor_label) else "motor-mendekati-orang-ROI-parkiran",
                                        "class" : "vehicle" if (objectA.name == motor_label) else "person",
                                        "daqID" : daqID,
                                        "camID" : camID,
                                        "filename": imageName
                                    }
                                    messagesQueue.append(data_result)
                                    omoDetected.append("motor-mendekati-orang-ROI-parkiran")
                                    omoLastSend = totalFrames
            if (time.time()-start_time_omo) > 30:
                start_time_omo = time.time()
                omoDetected = []

        ## ==================== END OF MOTOR MENDEKATI MOTOR ==================== ##
        ## DEMONSTRASI CODE ##
        if demonstrasi_detection:
            points = np.array(demonstrasi_roi, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            point_to_check = (int(cx), int(cy))
            result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)

            if result > 0 :
                if to.name == 'Mobil' :
                    vehicle_counted.append(to.objectID)
                    
                if to.name == 'Motor':
                    bike_counted.append(to.objectID)

                if to.name == 'Orang' :
                    people_counted.append(to.objectID)
                
                if len(people_counted) >= demonstrasi_people_limit or len(bike_counted) >= demonstrasi_bike_limit or len(vehicle_counted) >= demonstrasi_vehicle_limit :
                    if start_time_demonstrasi is None :
                        start_time_demonstrasi = time.time()
                    
                    elapsed_time_demonstrasi = time.time() - start_time_demonstrasi

                    if elapsed_time_demonstrasi > 10 :                    
                        data_result = {
                            "daqId": daqID,
                            "msgType": "OBD",
                            "camId": camID,
                            "time": helper.timestamp('%Y-%m-%d %H:%M:%S'),
                            "objects": [
                                {
                                    "id": "kepadatan-kendaraan-orang",
                                    "class": "object",
                                    "type": to.name
                                    # **context_config.get(object_context, {})
                                }
                            ],
                            "filename": imageName
                        }
                        messagesQueue.append(data_result)
                        start_time_demonstrasi = time.time()

        ## GERAKAN MENCURIGAKAN CODE ##
        if gerakan_mencurigakan_detection:
            points = np.array(gerakan_mencurigakan_roi, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)            
            point_to_check = (int(cx), int(cy))
            result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)

            if result > 0 :
                if not hasattr(to, 'time') or to.time is None:
                                to.time = time.time() 
                if to.name == 'Orang' :
                    # if to.isStill : 
                    #     to.time = time.time()
                    if (time.time() - to.time) > gerakan_mencurigakan_time_limit :
                        imageName2 = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-GM.jpg"
                        
                        frameGM = frame.copy()
                        cv2.rectangle(frameGM, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        _, buffer = cv2.imencode('.jpg', frameGM)
                        image_bytes = buffer.tobytes()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        data_result = {
                            "daqId": daqID,
                            "msgType": "OBD",
                            "camId": camID,
                            "time": helper.timestamp('%Y-%m-%d %H:%M:%S'),
                            "objects": [
                                {
                                    "id": "orang-ROI-waktu",
                                    "class": "person",
                                    "type": to.name,
                                    "key": int(to.objectID),
                                    "state": 1
                                }
                            ],
                            "filename": imageName2.replace(".jpg", f"-{messageCounter}.jpg")
                        }

                        #data_executor.submit(send_data, image_base64, imageName2.replace(".jpg", f"-{messageCounter}.jpg"), data_result, "jababeka-cc/sem")
                        messageCounter += 1

                        to.time = time.time()

        ## PKL CODE ##
        if pkl_detection:
            points = np.array(polygon_roi_pkl, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            point_to_check = (int(cx), int(cy))
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
                            imageNamePKL = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-PKL.jpg"

                            framePKL = frame.copy()
                            cv2.rectangle(framePKL, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            _, buffer = cv2.imencode('.jpg', framePKL)
                            image_bytes = buffer.tobytes()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            data_result = {
                                    "daqId": daqID,
                                    "msgType": "OBD",
                                    "camId": camID,
                                    "time": helper.timestamp('%Y-%m-%d %H:%M:%S'),
                                    "objects": [
                                        {
                                            "id": "PKL-ROI-waktu",
                                            "class": "vehicle",
                                            "type": to.name,
                                            "key": int(to.objectID),
                                            "state": 1
                                        }
                                    ],
                                    "file": imageNamePKL.replace(".jpg", f"-{messageCounter}.jpg")
                                }
                            
                            #data_executor.submit(send_data, image_base64, imageNamePKL.replace(".jpg", f"-{messageCounter}.jpg"), data_result, "jababeka-cc/sem")
                            messageCounter += 1

                            to.time = time.time()
                            to.status = True
                                
                else :
                    if to.state == 1:
                        to.state = 0
                        to.time = None
                        if to.status == True:
                            imageNamePKL = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-PKL.jpg"

                            framePKL = frame.copy()
                            cv2.rectangle(framePKL, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            _, buffer = cv2.imencode('.jpg', framePKL)
                            image_bytes = buffer.tobytes()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            data_result = {
                                    "daqId": daqID,
                                    "msgType": "OBD",
                                    "camId": camID,
                                    "time": helper.timestamp('%Y-%m-%d %H:%M:%S'),
                                    "objects": [
                                        {
                                            "id": "PKL-ROI-waktu",
                                            "class": "vehicle",
                                            "type": to.name,
                                            "key": int(to.objectID),
                                            "state": 0
                                        }
                                    ],
                                    "file": imageNamePKL.replace(".jpg", f"-{messageCounter}.jpg")
                                }
                            
                            #data_executor.submit(send_data, image_base64, imageNamePKL.replace(".jpg", f"-{messageCounter}.jpg"), data_result, "jababeka-cc/sem")
                            messageCounter += 1

                            to.time = time.time()
                            to.status = False
                            
        ## PARKIR LIAR CODE ##
        if parkir_liar_detection and to.name not in ['Orang', 'PKL', 'No object']:
            imageNamePL = f"{daqID}-{camID}-{helper.timestamp('%Y%m%d%H%M%S')}-PL.jpg"
            points = np.array(polygon_roi, np.int32)
            points = points.reshape((-1, 1, 2))
            point_to_check = (int(cx), int(cy))
            result = cv2.pointPolygonTest(points, point_to_check, measureDist=False)
            
            if result >= 0 :
                if to.state == 0 :
                    to.state = 1
            
                    if not hasattr(to, 'time') or to.time is None:
                        to.time = time.time()

                if to.state == 1 :
                    if (time.time() - to.time) >= parkir_liar_object_time_limit :
                        to.status = True

                        framePL = frame.copy()

                        cv2.rectangle(framePL, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        _, buffer = cv2.imencode('.jpg', framePL)
                        image_bytes = buffer.tobytes()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        upload_success = helper.uploadFile(image_base64, imageNamePL)

                        data_result = {
                            "colorRecognition" : True,
                            "alpr" : True,
                            "objectData" : to.name,
                            "objectID" : int(to.objectID),
                            "state" : int(to.state),
                            "bbox" : [int(x) for x in to.bbox[-1]],
                            "context" : "kendaraan-ROI-bahujalan-waktu",
                            "class" : "vehicle",
                            "daqID" : daqID,
                            "camID" : camID,
                            "filename": imageNamePL.replace(".jpg", f"-{messageCounter}.jpg")
                        }

                        #data_executor.submit(send_data, image_base64, imageNamePL.replace(".jpg", f"-{messageCounter}.jpg"), data_result)
                        messageCounter += 1
                        to.time = time.time()

            if result < 0 :
                if to.state == 1 :
                    to.state = 0
                    if to.state == 0 :
                        if to.status == True:
                            to.status = False

                            _, buffer = cv2.imencode('.jpg', frame)
                            image_bytes = buffer.tobytes()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                            data_result = {
                                "colorRecognition" : True,
                                "alpr" : True,
                                "objectData" : to.name,
                                "bbox": [int(x) for x in to.bbox[-1]],
                                "state" : int(to.state),
                                "objectID" : int(to.objectID),
                                "context" : "kendaraan-ROI-bahujalan-waktu",
                                "class" : "vehicle",
                                "daqID" : daqID,
                                "camID" : camID,
                                "filename": imageNamePL.replace(".jpg", f"-{messageCounter}.jpg")
                            }

                            #data_executor.submit(send_data, image_base64, imageNamePL.replace(".jpg", f"-{messageCounter}.jpg"), data_result)
                            messageCounter += 1

    if len(messagesQueue) > 0:
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        #for message in messagesQueue:
        #    if "context" in message:
        #        data_executor.submit(send_data, image_base64, imageName.replace(".jpg", f"-{messageCounter}.jpg"), message)
        #    else:
        #        data_executor.submit(send_data, image_base64, imageName.replace(".jpg", f"-{messageCounter}.jpg"), message, "jababeka-cc/sem")
        #   messageCounter += 1
        messagesQueue = []

    totalFrames += 1
    return frame  # Mengembalikan frame yang telah diproses

def process_cctv_stream(config):
    """Process video stream with multi-threaded object detection."""
    verbose = config["config_local"]["verbose"]
    verbose_image = config["config_local"]["verbose_image"]
    stream = config["config_local"]["stream"]
    zmq_address = config["config_api"]["zmq_address"]

    logger.info("Starting the video...") if verbose else None

    vs = cv2.VideoCapture(config["config_api"]["rtsp_url"])

    if not vs.isOpened():
        logger.error(f"Error: Unable to open stream {config['config_api']['rtsp_url']}")
        return
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        while True:
            try:
                has_frame, frame = vs.read()
                
                if not has_frame or frame is None:
                    logger.warning("Frame is empty or invalid. Reconnecting...")
                    vs.release()
                    vs = cv2.VideoCapture(config['config_api']['rtsp_url'])
            except cv2.error as e:
                helper.writelog(f"OpenCV: {e}")
                continue
            
            future = executor.submit(process_detections, frame, config)

            try:
                processed_frame = future.result(timeout=1)
                if processed_frame is not None:
                    cv2.imshow("Debugging", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                logger.error(f"Error in processing frame: {e}")
            
            metrics = metric_executor.submit(save_metric, sensor_id)
    
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with YOLO")
    parser.add_argument("--sensor_id", required=True, help="Sensor ID for configuration")
    args = parser.parse_args()

    sensor_id = args.sensor_id

    config = helper.loadConfig(sensor_id)
    verbose = config["config_local"]["verbose"]
    model = config['config_local']['deployment']
    daqID = config["config_device"]["daqID"]
    camID = config["config_device"]["camID"]

    offline_deployment = YOLO(model)

    process_cctv_stream(
        config
    )
