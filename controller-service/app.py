import time
import sys
sys.path.append('/home/refky/websocket/devKCI/geti_code/')
import config
import requests
import paho.mqtt.client as mqttClient
import json
import logging
import zipfile
import os
from minio import Minio
import shutil
import helper

with open("config.json", "r") as jsonfile:
    data = json.load(jsonfile)

with open("../file-id.txt", "r") as sensor_file:
    sensor_ids = sensor_file.read().splitlines()

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler(data['log']['filename'])  # Ganti 'logfile.txt' dengan nama file yang diinginkan

# Tingkat log yang akan ditangani oleh file handler
file_handler.setLevel(logging.INFO)

# Format log yang akan ditulis ke dalam file
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Menambahkan handler ke logger
logger.addHandler(file_handler)

def downloadDeployment(config_data):
    minio_client = Minio(config_data['minioEndPoint'], config_data['minioAccessKey'], config_data['minioSecretKey'], secure=False)
    bucket_name = "jpo-geti"
    # zip_filename = config_data['object']
    zip_filename = "Deployment-JPO People detectio.zip"
    folderdir = zip_filename.replace(".zip", "")

    os.chmod('/home/refky/websocket/devKCI/geti_code/', 0o777)

    try:
        minio_client.fget_object(bucket_name, zip_filename, f'../{zip_filename}')
        logger.info(f'File {zip_filename} berhasil di download dari {bucket_name}')
    except Exception as e:
        helper.send_telegram_message("JPO",sensor_id, e)
        logger.error(e)

    if os.path.exists("../" + folderdir):
        shutil.rmtree("../" + folderdir)

    with zipfile.ZipFile("../"+zip_filename, 'r') as zip_ref:
        zip_ref.extractall("../"+folderdir)
    logger.info(f"Deployment {zip_filename} unzipped")

def updateConfig(config_data, sensor_id):
    path = f"/home/refky/websocket/devKCI/geti_code/config-{sensor_id}.json"
    try:
        with open(path, "r") as file:
            config = json.load(file)
        config['config_api']['detection_id'] = config_data['sensor_id']
        logger.info(config_data['sensor_id'])
        config['config_api']['zmq_address'] = config_data['zmq_address']
        config['config_api']['rtsp_url'] = config_data['cam_stream_url']
        with open(path, "w") as file:
            json.dump(config, file, indent=4)
        helper.restart_script(sensor_id)
    except Exception as e:
        helper.send_telegram_message(e, "Change config failed")
        logger.error(e)
    logger.info("Change config successful")


def onConnect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connection successful")
    else:
        helper.send_telegram_message(rc, "Connection Failed")
        logger.error(f"Connection failed with return code: {rc}")

def onMessage(client, userdata, message):
    try:
        logger.info("Received a message")
        payload = message.payload.decode("utf-8")
        data = json.loads(payload)
        config_data = data['config'][0]
        try:
            if 'type' in config_data and config_data['type'] == "config":
                updateConfig(config_data, config_data['id'])
            elif 'type' in config_data and config_data['type'] == "geti-deployment":
                downloadDeployment(config_data)
        except Exception as e:
            logger.error(e)
    except Exception as e:
        logger.error(e)

# def onDisconnect(client, userdata, flags, rc):
#     if rc != 0:
#         logger.info("Trying to reconnect")
#     else:
#         logger.info("Connection successful")

if __name__ == "__main__":
    #Initiation MQTT
    hostAddress = data['config_mqtt']['MQTT_HOST']
    hostUsername = data['config_mqtt']['MQTT_USERNAME']
    hostPassword = data['config_mqtt']['MQTT_PASSWORD']
    portAddress = data['config_mqtt']['MQTT_PORT']
    baseTopicAddress = data['config_mqtt']['MQTT_TOPIC']


    print("[INFO] starting process...")
    for sensor_id in sensor_ids:
        topicAddress = f"{baseTopicAddress}{sensor_id}"
        client = mqttClient.Client("Python")
        client.username_pw_set(hostUsername, password=hostPassword)
        client.on_connect = onConnect
        client.on_message = onMessage
        client.connect(hostAddress, port=portAddress)
        client.subscribe(topicAddress)
        client.loop_forever()
        logger.info(f"Client subscribed to topic : {topicAddress}")
    