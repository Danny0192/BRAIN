import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import traceback
import os
from datetime import datetime

ruta_guardado = "C:\\Users\\Danny\\Desktop\\Brain\\BRAIN"
os.makedirs(ruta_guardado, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
archivo_txt = os.path.join(ruta_guardado, f"posiciones_{timestamp}.txt")
archivo_video = os.path.join(ruta_guardado, f"video_{timestamp}.mp4")

keypoints_of_interest = [7, 8, 13, 14, 15, 16]
keypoint_names = {7: "Codo_Izq", 8: "Codo_Der", 13: "Rodilla_Izq", 14: "Rodilla_Der", 15: "Tobillo_Izq", 16: "Tobillo_Der"}

with open(archivo_txt, mode='w', encoding='utf-8') as archivo:
    archivo.write("Timestamp,Keypoint_ID,Keypoint_Nombre,X_cm,Y_cm,Z_cm\n")

print("Inicializando YOLO...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n-pose.pt').to(device)
print("Modelo YOLO cargado en:", device)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align = rs.align(rs.stream.color)

frame_size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps_video = 16
video_writer = cv2.VideoWriter(archivo_video, fourcc, fps_video, frame_size)

def agregar_linea_txt(archivo, datos):
    with open(archivo, mode='a', encoding='utf-8') as f:
        f.write(','.join(map(str, datos)) + '\n')

def pixel_to_cm(pixel_x, pixel_y, depth_value, intrinsics):
    x = (pixel_x - intrinsics.ppx) / intrinsics.fx * depth_value
    y = (pixel_y - intrinsics.ppy) / intrinsics.fy * depth_value
    z = depth_value
    return x * 100, y * 100, z * 100

print("Esperando 2 segundos para iniciar cámara...")
time.sleep(2)

mostrar_vista = True
frame_count = 0
print("\033c", end="")
print("=== CAPTURA EN TIEMPO REAL ===")
print(f"Guardando en: {archivo_txt}\nPresiona 'q' para salir\n")

tiempo_inicio = time.time()

try:
    while True:
        start_time = time.time()

        try:
            frames = pipe.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                print("[WARN] Frame de profundidad o color no disponible")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display_image = color_image.copy()

            with torch.inference_mode():
                results = model(color_image, conf=0.6)

            deteccion = False

            for result in results:
                if result.keypoints is None or len(result.keypoints) == 0:
                    print("[INFO] No se detectaron keypoints")
                    continue

                keypoints = result.keypoints.xy.cpu().numpy().astype(int)[0]
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                for idx, (x, y) in enumerate(keypoints):
                    if idx not in keypoints_of_interest or not (0 <= x < 640 and 0 <= y < 480):
                        continue
                    depth = depth_frame.get_distance(x, y)
                    if depth == 0:
                        continue
                    x_cm, y_cm, z_cm = pixel_to_cm(x, y, depth, depth_intrinsics)
                    keypoint_name = keypoint_names.get(idx, f"Keypoint_{idx}")
                    agregar_linea_txt(archivo_txt, [current_time_str, idx, keypoint_name, f"{x_cm:.2f}", f"{y_cm:.2f}", f"{z_cm:.2f}"])

                    color = (0, 255, 0)
                    if idx in [7, 8]: color = (0, 0, 255)
                    elif idx in [13, 14]: color = (255, 0, 0)
                    elif idx in [15, 16]: color = (0, 255, 255)
                    cv2.circle(display_image, (x, y), 5, color, -1)
                    cv2.putText(display_image, keypoint_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.circle(color_image, (x, y), 3, color, -1)

                    deteccion = True

            if not deteccion:
                print("[INFO] No se encontraron keypoints válidos en este frame")

            video_writer.write(color_image)

            if mostrar_vista:
                cv2.imshow('Pose Detection', display_image)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("\nPrograma finalizado por el usuario")
                break

            frame_count += 1
            print(f"Retraso estimado: {(time.time() - start_time)*1000:.1f} ms", end='\r')

        except Exception as loop_err:
            print("[ERROR en bucle]:", loop_err)
            print(traceback.format_exc())

except Exception as e:
    print(f"\nError general: {e}\n{traceback.format_exc()}")

finally:
    if video_writer is not None:
        video_writer.release()
    pipe.stop()
    cv2.destroyAllWindows()
    print("\nRecursos liberados correctamente.")
