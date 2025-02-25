import cv2
import torch
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from collections import defaultdict
from ultralytics import YOLO

# Load mô hình YOLOv8
model = YOLO(r'D:/HT/thigiacmay/BaiTap/DemSoLuongXe/runs/detect/train3/weights/best.pt')

# Danh sách tên các class (phải khớp với mô hình)
class_names = ['Bus', 'O To', 'Xe May', 'Xe Tai']

# Trọng số cho từng loại phương tiện
vehicle_weights = {
    'Xe May': 0.5,
    'O To': 1,
    'Xe Tai': 2,
    'Bus': 2.5
}

# Hàm tính tổng lưu lượng xe
def calculate_traffic_flow(vehicle_counts):
    total_flow = sum(vehicle_counts[vehicle] * vehicle_weights.get(vehicle, 1) for vehicle in vehicle_counts)
    return total_flow

# Hàm xử lý ảnh
def process_image(file_path):
    img = cv2.imread(file_path)
    results = model(img)
    vehicle_counts = defaultdict(int)

    # Nhận diện và đếm số lượng xe
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            if cls < len(class_names):
                label = f"{class_names[cls]} {conf:.2f}"
                vehicle_counts[class_names[cls]] += 1

                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính lưu lượng xe
    total_flow = calculate_traffic_flow(vehicle_counts)

    # Hiển thị kết quả trong Tkinter
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    label_img.config(image=img)
    label_img.image = img

    count_text = "/n".join([f"{vehicle}: {count}" for vehicle, count in vehicle_counts.items()])
    label_count.config(text=f"Kết quả đếm xe:/n{count_text}/nLưu lượng xe: {total_flow:.2f}")

# Hàm xử lý video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        vehicle_counts = defaultdict(int)

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                if conf > 0.5 and cls < len(class_names):
                    label = f"{class_names[cls]} {conf:.2f}"
                    vehicle_counts[class_names[cls]] += 1

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

        # Tính lưu lượng xe
        total_flow = calculate_traffic_flow(vehicle_counts)

        # Hiển thị kết quả trong Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(frame_pil)

        label_img.config(image=frame_tk)
        label_img.image = frame_tk

        count_text = "/n".join([f"{vehicle}: {count}" for vehicle, count in vehicle_counts.items()])
        label_count.config(text=f"Kết quả đếm xe:/n{count_text}/nLưu lượng xe: {total_flow:.2f}")

        root.update()

    cap.release()

# Hàm để chọn tệp (ảnh hoặc video)
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("All Files", "*.*"), ("Image files", "*.jpg;*.png;*.jpeg"), ("Video files", "*.mp4;*.avi;*.mov")]
    )

    if not file_path:
        return

    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file_path)
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(file_path)

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Nhận Diện và Đếm Phương Tiện Giao Thông")

btn_select = tk.Button(root, text="Chọn Ảnh hoặc Video", command=select_file, font=("Arial", 12))
btn_select.pack(pady=10)

label_img = tk.Label(root)
label_img.pack()

label_count = tk.Label(root, text="Kết quả đếm xe sẽ hiển thị ở đây", font=("Arial", 12), fg="blue")
label_count.pack(pady=10)

root.mainloop()
