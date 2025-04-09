import time
import os
import config
import requests
import sys

from multiprocessing import Process

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.append('/home/rastekid/apps/filewatcher/')
sys.path.insert(0, '../')

import utils

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".jpg"):
            image_name = event.src_path.split('/')[-1]
            image_path = os.path.dirname(event.src_path)

            id_analytic = image_name.split("-")[0] + '-' + image_name.split("-")[1]
            _timestamp_ = image_name.split("-")[2].replace(".jpg", "")

            utils.send_mqtt(id_analytic, image_path, image_name, _timestamp_)


def file_watcher():
    observer = Observer()
    event_handler = MyHandler()

    with open(config.MAIN_PATH + config.FILE_ID, 'r') as file:
        id_from_file = []
        for line in file:
            id_from_file.append(line.strip())
            
    folders_to_watch = [config.MAIN_PATH + 'tmp-' + element for element in id_from_file]
    
    for folder in folders_to_watch:
        observer.schedule(event_handler, path=folder, recursive=True)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


# def get_config():
#     auth_token = auth_login()
#     duration_get_config = config.DURATION_GET_CONFIG

#     while True:
#         with open(config.MAIN_PATH + config.FILE_ID, "r") as a_file:
#             a_file = [line.strip() for line in a_file]
        
#         for sensor_id in a_file:    
#             url_sensor = config.BACKEND_ENDPOINT + 'master/api/sensor'
            
#             headers = {'Authorization': "Bearer " + auth_token}
#             params = {'id': sensor_id}

#             req = requests.get(url=url_sensor, params=params, headers=headers)
#             response = req.json()

#             if req.status_code == 200:
                
#                 url_config = config.MAIN_PATH + "config-" + sensor_id + ".json"
                
#                 # compare_config(response["data"][0], url_config)
#                 duration_get_config = config.DURATION_GET_CONFIG
#             elif req.status_code == 401:
#                 auth_token = auth_login()
#                 duration_get_config = 1
#                 break
#             else:
#                 print('error')
#                 break

#         time.sleep(duration_get_config)


def auth_login():
    urlApiLogin = config.BACKEND_ENDPOINT + 'account/auth/login'
    username = config.BACKEND_AUTH_USERNAME
    password = config.BACKEND_AUTH_PASSWORD
    grant_type = config.BACKEND_AUTH_GRANT_TYPE
    authToken = None

    response = requests.post(urlApiLogin, data={
                             'username': username, 'password': password, 'grant_type': grant_type})

    if response.status_code == 200:
        authToken = response.json()['access_token']
    else:
        authToken = ""

    return authToken


def compare_config(data_api, url_config):
    import json
    
    with open(url_config, "r") as json_file:
        data = json.load(json_file)

    data["config_api"]["detection_id"] = data_api["id"]
    data["config_api"]["analytic_id"] = data_api["analyticId"]
    data["config_api"]["cam_id"] = data_api["camId"]
    data["config_api"]["zmq_address"] = data_api["zmqAddress"]
    data["config_api"]["rtsp_url"] = data_api["detailCamera"]["url"]
    data["config_api"]["det_duration"] = data_api["detDuration"]
    data["config_api"]["confidence_val"] = data_api["confidenceVal"]
    data["config_api"]["skip_frame"] = data_api["skipFrame"]
    data["config_api"]["line_detection"] = data_api["lineDetection"]
    data["config_api"]["counting"] = data_api["counting"]

    with open(url_config, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    print("[INFO] starting process...")
    p_file_watcher = Process(target=file_watcher)
    # p_get_config = Process(target=get_config)

    p_file_watcher.start()
    # p_get_config.start()

    p_file_watcher.join()
    # p_get_config.join()
