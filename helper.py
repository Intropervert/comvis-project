import json
import datetime
import logging
import base64
import io
import paho.mqtt.client as mqtt
from minio import Minio
from minio.error import S3Error
import global_config
import psutil
import subprocess

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

minio_client = Minio(global_config.MINIO_ENDPOINT, global_config.MINIO_ACCESS_KEY, global_config.MINIO_SECRET_KEY, secure=False)
client = mqtt.Client()
client.username_pw_set(global_config.MQTT_USERNAME, global_config.MQTT_PASSWORD)
code = client.connect(global_config.MQTT_HOST, global_config.MQTT_PORT, 30)
client.loop_start()

def send_mqtt(msg, topic=global_config.MQTT_TOPIC):
    try:
        if code == 0:
            client.publish(topic, json.dumps(msg), qos=2, retain=False)

    except Exception as e:
        print(e)
        return None

def uploadFile(imageBase64, filename):
    try:
        image_data = base64.b64decode(imageBase64)
        image_bytes = io.BytesIO(image_data)

        if not minio_client.bucket_exists(global_config.MINIO_BUCKET_NAME):
            minio_client.make_bucket(global_config.MINIO_BUCKET_NAME)

        minio_client.put_object(
            bucket_name=global_config.MINIO_BUCKET_NAME,
            object_name=filename,
            data=image_bytes,
            length=len(image_data),
            content_type="image/jpeg"
        )
        return True
    except S3Error as e:
        print("MinIO Error: " + str(e))
        return False
    except Exception as e:
        print(f"An exception occurred during file upload: {e}")
        return False

#### =================== LOAD CONFIG =================== ####
def loadConfig(sensor_id):
    config_path = global_config.MAIN_PATH + f"config-{sensor_id}.json"
    print("Full Path:", config_path)  # Debugging
    with open(config_path, "r") as file:
        return json.load(file)


#### =================== TIMESTAMP =================== ####
def timestamp(format):
    timestamp = datetime.datetime.now()
    formatted_timestamp = timestamp.strftime(format)
    return str(formatted_timestamp)


#### =================== WRITE LOG =================== ####
def writelog(sensor_id, msg):
    try:
        log_file = global_config.MAIN_PATH + "log-" + str(sensor_id) + ".txt"
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{current_datetime}] {msg}"

        # Write the log message to the file
        with open(log_file, "a") as file:
            file.write(log_msg + "\n")
    except Exception as e:
        logger.info(f"Error logging data: {str(e)}")

# Function to get the ping result
def get_ping(host):
    try:
        # Send a single ping to the host (e.g., google.com or an IP address)
        response = subprocess.run(['ping', '-c', '1', host], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if response.returncode == 0:
            # Parse the response time (ms) from the ping result
            ping_time = response.stdout.split('time=')[1].split(' ')[0]
            return f"{ping_time} ms"
        else:
            return "Ping failed"
    except Exception as e:
        return f"Error pinging: {str(e)}"

# Function to get CPU usage
def get_cpu_usage():
    # Get CPU usage percentage (using psutil)
    return psutil.cpu_percent(interval=1)

# Function to get memory usage
def get_memory_usage():
    # Get memory usage in percentage (using psutil)
    memory = psutil.virtual_memory()
    return memory.percent
