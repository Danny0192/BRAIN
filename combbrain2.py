import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import time
import traceback
from collections import deque
import csv
import time 
from datetime import datetime
from sort import Sort

 # Calcular velocidad
                    # current_velocity = np.zeros(3)
                    # if len(position_history[idx]) >= 2 and len(time_history[idx]) >= 2:
                    #     current_velocity = calculate_velocity(
                    #         position_history[idx][-1], 
                    #         position_history[idx][-2],
                    #         time_history[idx][-1],
                    #         time_history[idx][-2]
                    #     )
                    #     velocity_history[idx].append(current_velocity)
                    
                    # # Calcular aceleración
                    # current_acceleration = np.zeros(3)
                    # if len(velocity_history[idx]) >= 2 and len(time_history[idx]) >= 2:
                    #     current_acceleration = calculate_acceleration(
                    #         velocity_history[idx][-1],
                    #         velocity_history[idx][-2],
                    #         time_history[idx][-1],
                    #         time_history[idx][-2]
                    #     )
                    
                    # # Suavizar mediciones
                    # smoothed_position = smooth_measurement(position_history[idx])
                    # smoothed_velocity = smooth_measurement(velocity_history[idx]) if velocity_history[idx] else np.zeros(3)
                    
                    # # Calcular magnitudes
                    # vel_magnitude = np.sqrt(np.sum(smoothed_velocity**2))
                    # acc_magnitude = np.sqrt(np.sum(current_acceleration**2))
                    
                    # # Limitar valores extremos para visualización
                    # vel_magnitude = min(vel_magnitude, 200)  # cm/s
                    # acc_magnitude = min(acc_magnitude, 1000)  # cm/s²

                    # if idx in keypoints_of_interest:
                        # print(f"{keypoint_name}: ")
                        # print(f"  Pos: X={smoothed_position[0]:.1f}, Y={smoothed_position[1]:.1f}, Z={smoothed_position[2]:.1f} cm")
                        # print(f"  Vel: X={smoothed_velocity[0]:.1f}, Y={smoothed_velocity[1]:.1f}, Z={smoothed_velocity[2]:.1f} cm/s (Mag={vel_magnitude:.1f})")
                        # print(f"  Acc: X={current_acceleration[0]:.1f}, Y={current_acceleration[1]:.1f}, Z={current_acceleration[2]:.1f} cm/s² (Mag={acc_magnitude:.1f})")
                        # current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        # agregar_fila_csv(nombre_archivo, [current_time_str, idx, current_acceleration[0], current_acceleration[1], current_acceleration[2]])
try:
    tracker1 = Sort()
    nombre_archivo = 'datosss.csv'
    # Definir el dispositivo para YOLO
    print("Inicializando YOLO...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    model = YOLO('yolov8s-pose.pt').to(device)
    print("Modelo YOLO cargado correctamente")

    # Iniciar RealSense
    print("Inicializando RealSense...")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    print("Iniciando pipeline...")
    profile = pipe.start(cfg)
    print("Pipeline iniciado correctamente")

    # Obtener información de la cámara para convertir píxeles a unidades métricas
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Escala de profundidad: {depth_scale}")

    # Obtener los parámetros intrínsecos de la cámara
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"Parámetros intrínsecos obtenidos.")

    # Obtener objeto de alineación para alinear la profundidad con el color
    align = rs.align(rs.stream.color)

    position_history = {i: deque(maxlen=5) for i in range(17)}  # Historial de 5 posiciones
    time_history = {i: deque(maxlen=5) for i in range(17)}      # Historial de tiempos
    velocity_history = {i: deque(maxlen=4) for i in range(17)}  # Historial de velocidades
    
    # Definir nombres de los keypoints
    keypoint_names = {
        7: "Codo_Izq",
        8: "Codo_Der",
        13: "Rodilla_Izq",
        14: "Rodilla_Der",
        15: "Tobillo_Izq",
        16: "Tobillo_Der"
    }

    def agregar_fila_csv(nombre_archivo, datos):
        """Agrega una fila de datos a un archivo CSV."""
        with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(datos)

    # Función para convertir de píxeles a coordenadas 3D en centímetros
    def pixel_to_cm(pixel_x, pixel_y, depth_value, intrinsics):
        # Convertir de píxeles a metros
        x = (pixel_x - intrinsics.ppx) / intrinsics.fx * depth_value
        y = (pixel_y - intrinsics.ppy) / intrinsics.fy * depth_value
        z = depth_value
        # Convertir de metros a centímetros
        return x * 100, y * 100, z * 100
    
    # Función para calcular velocidad desde posiciones
    def calculate_velocity(pos_current, pos_previous, time_current, time_previous):
        if time_current == time_previous:
            return np.zeros(3)
        dt = time_current - time_previous
        return (pos_current - pos_previous) / dt
    
    # Función para calcular aceleración desde velocidades
    def calculate_acceleration(vel_current, vel_previous, time_current, time_previous):
        if time_current == time_previous:
            return np.zeros(3)
        dt = time_current - time_previous
        return (vel_current - vel_previous) / dt
    
    # Función para aplicar un filtro de suavizado a las mediciones
    def smooth_measurement(history_array):
        if len(history_array) < 2:
            return history_array[-1] if history_array else np.zeros(3)
        
        # Aplicar un filtro exponencial simple
        weights = np.array([0.6, 0.3, 0.1, 0.0, 0.0])[:len(history_array)]
        weights = weights / np.sum(weights)
        
        result = np.zeros(3)
        for i, item in enumerate(reversed(history_array)):
            if i >= len(weights):
                break
            result += item * weights[i]
            
        return result

    print("Esperando 2 segundos para que la cámara se inicialice...")
    time.sleep(2)
    
    print("Iniciando bucle principal...")
    frame_count = 0
    prev_time = None
    dt = 1/30.0  # Valor inicial
    
    # Limpiar la consola
    print("\033c", end="")
    print("=== SEGUIMIENTO DE ARTICULACIONES EN TIEMPO REAL ===")
    print("Posiciones (cm), Velocidades (cm/s) y Aceleraciones (cm/s²)")
    print("Presiona 'q' para salir")
    print("--------------------------------------------------")
    
    while True:
        try:
            start_time = time.time()
            
            frames = pipe.wait_for_frames()
            
            # Alinear los frames de profundidad al color
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Detección con YOLOv8 Pose
            results = model(color_image, conf=0.7)

            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            # Calcular dt real entre frames
            if prev_time is not None:
                dt = current_time - prev_time
            
            # Limpiar la zona de impresión
            if frame_count % 30 == 0:
                print("\033c", end="")
                print("=== SEGUIMIENTO DE ARTICULACIONES EN TIEMPO REAL ===")
                print("Posiciones (cm), Velocidades (cm/s) y Aceleraciones (cm/s²)")
                print("Presiona 'q' para salir")
                print("--------------------------------------------------")

            detection_found = False
            for result in results:
                if result.keypoints is None or len(result.keypoints) == 0:
                    continue
                
                detection_found = True
                keypoints_of_interest = [7, 8, 13, 14, 15, 16]
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Filtrar detecciones que pertenezcan a las clases de interés y tengan confianza > 0.25
                filtered_indices = np.where((np.isin(classes, keypoints_of_interest)) & (confidences > 0.25))[0]
                keyp = result.keypoints.xy.cpu().numpy()[filtered_indices].astype(int)
                # print(f"Frame: {frame_count} | Tiempo: {current_time:.2f}s | dt: {dt*1000:.1f}ms")
                # print("--------------------------------------------------")
                tracks = tracker1.update(keyp, classes)
                tracks = tracks.astype(int)
                for (x, y, idx) in enumerate(tracks):
                    if idx not in keypoints_of_interest:
                        continue
                        
                    if x < 0 or x >= 640 or y < 0 or y >= 480:
                        continue

                    # Obtener profundidad en metros
                    depth = depth_frame.get_distance(x, y)
                    if depth == 0:
                        continue

                    # Convertir a coordenadas 3D en centímetros
                    x_cm, y_cm, z_cm = pixel_to_cm(x, y, depth, color_intrinsics)
                    current_position = np.array([x_cm, y_cm, z_cm])
                    
                    # Registrar posición y tiempo
                    position_history[idx].append(current_position)
                    time_history[idx].append(current_time)
                    
                   
                    print(f"IDX:{idx}")
                    # Imprimir keypoints importantes
                    keypoint_name = keypoint_names.get(idx, f"Keypoint_{idx}")
                    print(keypoint_name)

                    # Dibujar keypoint base
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{idx}", (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if detection_found:
                print("--------------------------------------------------")
            
            prev_time = current_time

            # Mostrar imágenes
            cv2.imshow('Pose Detection', color_image)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
            cv2.imshow('Depth Image', depth_colormap)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("\nPrograma finalizado por el usuario")
                break
                
            frame_count += 1
            
            target_interval = 1 / 10.0 
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        except Exception as e:
            print(f"\nError en el bucle principal: {e}")
            print(traceback.format_exc())
            break


    print("Cerrando recursos...")
    pipe.stop()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente")

except Exception as e:
    print(f"Error general: {e}")
    print(traceback.format_exc())
    
    # Intentar liberar recursos en caso de error
    try:
        pipe.stop()
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    print("Programa terminado con errores")