import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import os
import traceback

try:
    # === CONFIGURA TU RUTA ===
    ruta_output = r"C:\\Users\\Danny\\Desktop\\Brain\\BRAIN"
    nombre_archivo = "video_keypoints.mp4"
    ruta_completa = os.path.join(ruta_output, nombre_archivo)

    if not os.path.exists(ruta_output):
        os.makedirs(ruta_output)

    # === CARGAR MODELO YOLO ===
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov8n-pose.pt').to(device)

    # === INICIAR CÁMARA REALSENSE ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    time.sleep(2)

    # === VARIABLES PARA TIEMPO Y VIDEO ===
    keypoints_of_interest = [7, 8, 13, 14, 15, 16]
    frame_size = (640, 480)
    frame_times = []
    frame_count = 0
    tiempo_inicio = time.time()
    tiempo_primer_frame = None
    tiempo_ultimo_frame = None
    proporcion_real_a_video = 0.80  # <== este valor ajusta la velocidad de grabación

    out = None  # Se inicializa después con el FPS corregido
    print("Grabando video... Presiona 'q' para detener.")

    while True:
        start_time = time.time()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # === DETECCIÓN YOLO ===
        results = model(color_image, conf=0.7)

        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                for idx, (x, y) in enumerate(keypoints[0]):
                    if idx in keypoints_of_interest and 0 <= x < 640 and 0 <= y < 480:
                        size = 8
                        color = (0, 255, 0)
                        top_left = (x - size // 2, y - size // 2)
                        bottom_right = (x + size // 2, y + size // 2)
                        cv2.rectangle(color_image, top_left, bottom_right, color, thickness=-1)

        # === MEDIR TIEMPO ENTRE FRAMES ===
        now = time.time()
        frame_times.append(now)
        if len(frame_times) > 30:
            frame_times.pop(0)

        # Guardar tiempo inicial y final para duración real
        if tiempo_primer_frame is None:
            tiempo_primer_frame = now
        tiempo_ultimo_frame = now

        # === CREAR VIDEOWRITER CON FPS REAL CORREGIDO ===
        if out is None and len(frame_times) >= 2:
            fps_real = len(frame_times) / (frame_times[-1] - frame_times[0])
            fps_corr = fps_real * proporcion_real_a_video
            fps_corr = max(5, min(fps_corr, 30))  # evitar valores extremos
            print(f"FPS real: {fps_real:.2f} | FPS corregido: {fps_corr:.2f}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ruta_completa, fourcc, fps_corr, frame_size)

        # === MOSTRAR Y GUARDAR FRAME ===
        cv2.imshow("Keypoints seleccionados", color_image)
        if out:
            out.write(color_image)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Grabación detenida por el usuario.")
            break

    # === FINALIZAR ===
    duracion_real = tiempo_ultimo_frame - tiempo_primer_frame if tiempo_primer_frame else 0
    duracion_video = frame_count / fps_corr if fps_corr else 0

    print("\n Video guardado correctamente.")
    print(f"Ruta: {ruta_completa}")
    print(f"Duración real: {duracion_real:.2f} s")
    print(f"Duración de video: {duracion_video:.2f} s")
    print(f"Frames grabados: {frame_count}")
    print(f"FPS real estimado: {fps_corr:.2f}")

    if out:
        out.release()
    pipeline.stop()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc())
    try:
        pipeline.stop()
    except:
        pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
