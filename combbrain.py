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
try:
    nombre_archivo = 'datosss.csv'
    # Definir el dispositivo para YOLO
    print("Inicializando YOLO...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    model = YOLO('yolov8n-pose.pt').to(device)
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

    # Inicializar Kalman Filters para cada keypoint con modelo de aceleración constante
    def create_kalman_filter():
        # Modelo de 9 estados: x, y, z, vx, vy, vz, ax, ay, az
        kf = KalmanFilter(dim_x=9, dim_z=3)
        
        # Matriz de transición de estado con aceleración
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0, 0.5, 0, 0],    # x = x + vx*dt + 0.5*ax*dt^2
            [0, 1, 0, 0, 1, 0, 0, 0.5, 0],    # y = y + vy*dt + 0.5*ay*dt^2
            [0, 0, 1, 0, 0, 1, 0, 0, 0.5],    # z = z + vz*dt + 0.5*az*dt^2
            [0, 0, 0, 1, 0, 0, 1, 0, 0],      # vx = vx + ax*dt
            [0, 0, 0, 0, 1, 0, 0, 1, 0],      # vy = vy + ay*dt
            [0, 0, 0, 0, 0, 1, 0, 0, 1],      # vz = vz + az*dt
            [0, 0, 0, 0, 0, 0, 1, 0, 0],      # ax = ax
            [0, 0, 0, 0, 0, 0, 0, 1, 0],      # ay = ay
            [0, 0, 0, 0, 0, 0, 0, 0, 1]       # az = az
        ])
        
        # Matriz de medición: solo medimos posición, no velocidad ni aceleración
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        
        # Incertidumbre inicial alta
        kf.P *= 1000
        
        # Ruido de medición
        kf.R = np.eye(3) * 0.5
        
        # Ruido de proceso
        # Matriz de covarianza del ruido del proceso más elaborada
        # Mayor en componentes de aceleración, menor en posición
        kf.Q = np.eye(9)
        kf.Q[0:3, 0:3] *= 0.01    # Posición
        kf.Q[3:6, 3:6] *= 0.1     # Velocidad
        kf.Q[6:9, 6:9] *= 1.0     # Aceleración
        
        # Inicializar con valores neutros
        kf.x = np.zeros(9)
        
        return kf

    kalman_filters = {i: create_kalman_filter() for i in range(17)}
    kalman_initialized = {i: False for i in range(17)}  # Trackear si el filtro ha sido inicializado
    
    # Variables para calcular velocidad y aceleración
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
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                
                print(f"Frame: {frame_count} | Tiempo: {current_time:.2f}s | dt: {dt*1000:.1f}ms")
                print("--------------------------------------------------")
                
                keypoints_of_interest = [7, 8, 13, 14, 15, 16]

                for idx, (x, y) in enumerate(keypoints[0]):
                    if idx not in keypoints_of_interest:
                        continue
                        
                    # Verificar límites válidos
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
                    
                    # Calcular velocidad (diferencias directas)
                    current_velocity = np.zeros(3)
                    if len(position_history[idx]) >= 2 and len(time_history[idx]) >= 2:
                        current_velocity = calculate_velocity(
                            position_history[idx][-1], 
                            position_history[idx][-2],
                            time_history[idx][-1],
                            time_history[idx][-2]
                        )
                        velocity_history[idx].append(current_velocity)
                    
                    # Calcular aceleración (diferencias directas)
                    current_acceleration = np.zeros(3)
                    if len(velocity_history[idx]) >= 2 and len(time_history[idx]) >= 2:
                        current_acceleration = calculate_acceleration(
                            velocity_history[idx][-1],
                            velocity_history[idx][-2],
                            time_history[idx][-1],
                            time_history[idx][-2]
                        )
                    
                    # Suavizar mediciones para reducir ruido
                    smoothed_position = smooth_measurement(position_history[idx])
                    smoothed_velocity = smooth_measurement(velocity_history[idx]) if velocity_history[idx] else np.zeros(3)
                    
                    # Actualizar Kalman filter
                    kalman = kalman_filters[idx]
                    
                    # Si es la primera vez que vemos este keypoint, inicializar el filtro
                    if not kalman_initialized[idx]:
                        kalman.x[0:3] = current_position  # Posición
                        if len(velocity_history[idx]) > 0:
                            kalman.x[3:6] = current_velocity  # Velocidad
                        if len(velocity_history[idx]) > 1:
                            kalman.x[6:9] = current_acceleration  # Aceleración
                        kalman_initialized[idx] = True
                        continue
                    
                    # Actualizar la matriz de transición con el dt real
                    dt2 = dt * dt
                    kalman.F[0, 3] = dt      # Actualizar x con vx*dt
                    kalman.F[0, 6] = 0.5*dt2 # Actualizar x con 0.5*ax*dt^2
                    kalman.F[1, 4] = dt      # Actualizar y con vy*dt
                    kalman.F[1, 7] = 0.5*dt2 # Actualizar y con 0.5*ay*dt^2
                    kalman.F[2, 5] = dt      # Actualizar z con vz*dt
                    kalman.F[2, 8] = 0.5*dt2 # Actualizar z con 0.5*az*dt^2
                    kalman.F[3, 6] = dt      # Actualizar vx con ax*dt
                    kalman.F[4, 7] = dt      # Actualizar vy con ay*dt
                    kalman.F[5, 8] = dt      # Actualizar vz con az*dt
                    
                    # Predecir
                    kalman.predict()
                    
                    # Actualizar con la medición
                    kalman.update(current_position)
                    
                    # Obtener estado estimado
                    estimated_state = kalman.x
                    
                    # Extraer valores filtrados
                    x_filtered, y_filtered, z_filtered = estimated_state[0:3]
                    vx, vy, vz = estimated_state[3:6]
                    ax, ay, az = estimated_state[6:9]
                    
                    # Mezclar con cálculos directos para mejorar respuesta
                    alpha_v = 0.3  # Factor para velocidad
                    alpha_a = 0.5  # Factor para aceleración (mayor peso a mediciones directas)
                    
                    # Mezclar velocidades (Kalman + directas)
                    vx = vx * (1-alpha_v) + smoothed_velocity[0] * alpha_v
                    vy = vy * (1-alpha_v) + smoothed_velocity[1] * alpha_v
                    vz = vz * (1-alpha_v) + smoothed_velocity[2] * alpha_v
                    
                    # Mezclar aceleraciones (Kalman + directas)
                    if len(velocity_history[idx]) >= 2:
                        ax = ax * (1-alpha_a) + current_acceleration[0] * alpha_a
                        ay = ay * (1-alpha_a) + current_acceleration[1] * alpha_a
                        az = az * (1-alpha_a) + current_acceleration[2] * alpha_a
                    
                    # Actualizar estado con valores mezclados
                    kalman.x[3:6] = np.array([vx, vy, vz])
                    kalman.x[6:9] = np.array([ax, ay, az])
                    
                    # Calcular magnitudes
                    vel_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
                    acc_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
                    
                    # Limitar valores extremos para visualización
                    vel_magnitude = min(vel_magnitude, 200)  # cm/s
                    acc_magnitude = min(acc_magnitude, 1000)  # cm/s²
                    
                    # Imprimir keypoints importantes
                    keypoint_name = keypoint_names.get(idx, f"Keypoint_{idx}")
                    if idx in keypoints_of_interest:
                        print(f"{keypoint_name}: ")
                        print(f"  Pos: X={x_filtered:.1f}, Y={y_filtered:.1f}, Z={z_filtered:.1f} cm")
                        print(f"  Vel: X={vx:.1f}, Y={vy:.1f}, Z={vz:.1f} cm/s (Mag={vel_magnitude:.1f})")
                        print(f"  Acc: X={ax:.1f}, Y={ay:.1f}, Z={az:.1f} cm/s² (Mag={acc_magnitude:.1f})")
                        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Hasta milisegundos
                        agregar_fila_csv(nombre_archivo, [current_time_str, idx, ax, ay, az])

                    # Dibujar keypoint base
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{idx}", (x + 5, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Codificar la velocidad en el color (velocidad alta = rojo)
                    max_speed = 100.0  # cm/s
                    speed_ratio = min(vel_magnitude / max_speed, 1.0)
                    color_b = int(255 * (1 - speed_ratio))
                    color_g = int(255 * (1 - speed_ratio))
                    color_r = int(255 * speed_ratio)

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
            
            # Controlar la velocidad de impresión
            # Controlar la frecuencia de muestreo (ej. 50 Hz = 0.02 s)
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