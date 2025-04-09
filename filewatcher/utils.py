import config
import datetime
import time


def send_mqtt(id_analytic, image_path, image_name, _timestamp_):
    import paho.mqtt.client as mqtt
    import json

    print()
    print("MQTT Start")

    broker_address = config.MQTT_HOST
    port = config.MQTT_PORT
    username = config.MQTT_USERNAME
    password = config.MQTT_PASSWORD
    topic_pub = config.MQTT_TOPIC

    log_file = image_path.replace("/tmp", "/log") + ".txt"

    _timestamp_ = datetime.datetime.strptime(_timestamp_, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    _value_ = None

    try:
        client = mqtt.Client()
        client.username_pw_set(username, password)
        code = client.connect(broker_address, port, 30)

        ### counting ###
        time.sleep(0.5)
        if 'PPLCNT' in id_analytic:
            with open(log_file, "r") as file:
                lines = file.readlines()

            last_data = lines[-1].strip()
            parts = last_data.split(' : ')
            _value_ = parts[1].strip()
        
        ### detection ###
        else:
            _value_ = "1"

        msg = [{
            "id": id_analytic,
            "time": _timestamp_,
            "value": _value_
        }]

        print(msg)

        if code == 0:
            print("MQTT Connected successfully ")

            client.loop_start()
            while True:
                client.publish(topic_pub, json.dumps(msg), qos=2, retain=False)
                print("MQTT message sent")
                client.disconnect()

                # if not 'PPLCNT' in id_analytic:
                uploadFile( (image_path + '/' + image_name), image_name )

                break

        else:
            print("MQTT Bad connection. Code: ", code)
            client.disconnect()

    except Exception as e:
        print(f"MQTT An exception occurred: {e}")
        return None

    ### clear folder ###
    clearFolder(image_path)
    # clearLog(log_file)


def uploadFile(local_file_path, remote_file_name):
    from minio import Minio
    from minio.error import S3Error

    minio_endpoint = config.MINIO_ENDPOINT
    access_key = config.MINIO_ACCESS_KEY
    secret_key = config.MINIO_SECRET_KEY
    bucket_name = config.MINIO_BUCKET_NAME

    minio_client = Minio(minio_endpoint, access_key, secret_key, secure=False)

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        minio_client.fput_object(
            bucket_name, remote_file_name, local_file_path)

        print(f"File {remote_file_name} uploaded successfully to {bucket_name}")

    except S3Error as e:
        print(f"Error: {e}")


def clearFolder(folder_path):
    import os

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                os.remove(file_path)


def clearLog(file_path):
    with open(file_path, 'w') as file:
        file.truncate(0)
