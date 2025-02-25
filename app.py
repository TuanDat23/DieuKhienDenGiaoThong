from flask import Flask, render_template, jsonify, request
import paho.mqtt.client as mqtt
import base64
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
latest_image_path = "static/latest_image.jpg"
traffic_state = {"light": "red", "time": 10}

# Khởi tạo mô hình YOLO
model = YOLO("yolov8n.pt")

def save_image_from_mqtt(payload):
    try:
        img_data = base64.b64decode(payload)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(latest_image_path, img)
    except Exception as e:
        print("Lỗi khi lưu ảnh từ MQTT:", e)

# Xử lý ảnh khi đèn vàng bật
def analyze_image():
    try:
        img = cv2.imread(latest_image_path)
        results = model(img)
        count = len(results[0].boxes)  # Đếm số phương tiện
        print(f"Số phương tiện nhận diện: {count}")
        
        # Cập nhật thời gian đèn dựa vào số lượng xe
        global traffic_state
        if count < 3:
            traffic_state["time"] = 10
        elif count < 6:
            traffic_state["time"] = 15
        else:
            traffic_state["time"] = 20

        # Lưu ảnh nhận diện
        for r in results:
            img = r.plot()
        cv2.imwrite(latest_image_path, img)
    except Exception as e:
        print("Lỗi khi phân tích ảnh:", e)

# Xử lý MQTT
mqtt_broker = "192.168.0.115"
topic = "img"

def on_connect(client, userdata, flags, rc):
    print("Kết nối MQTT thành công!")
    client.subscribe(topic)

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    if payload == "end":
        print("Ảnh nhận diện hoàn tất!")
    else:
        save_image_from_mqtt(payload)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(mqtt_broker, 1883, 60)
mqtt_client.loop_start()

# Điều khiển đèn giao thông
def update_traffic_light():
    global traffic_state
    current_time = traffic_state["time"]
    if current_time > 0:
        traffic_state["time"] -= 1
    else:
        if traffic_state["light"] == "red":
            traffic_state["light"] = "green"
            traffic_state["time"] = 15
        elif traffic_state["light"] == "green":
            traffic_state["light"] = "yellow"
            traffic_state["time"] = 3
        elif traffic_state["light"] == "yellow":
            analyze_image()
            traffic_state["light"] = "red"
            # Thời gian đèn đỏ đã được tính từ phân tích ảnh
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/traffic_state")
def get_traffic_state():
    update_traffic_light()
    return jsonify(traffic_state)

@app.route("/analyze_image", methods=["POST"])
def analyze():
    analyze_image()
    return jsonify({"status": "done"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
