import csv
from datetime import datetime
import math
import numpy as np
promedios_x = []
promedios_y = []
promedios_z = []
counter = 0
def convertir_csv_a_txt(archivo_csv, archivo_txt):
    """Lee un archivo CSV y lo guarda en un archivo TXT con el mismo formato."""
    with open(archivo_csv, mode='r', newline='', encoding='utf-8') as csv_file, \
         open(archivo_txt, mode='w', encoding='utf-8') as txt_file:
        lector = csv.reader(csv_file)
        for fila in lector:
            txt_file.write(','.join(fila) + '\n')

def encontrar_tiempos_similares(archivo_csv, archivo_txt, umbral_segundos=1):
    """Compara los resultados del CSV y del TXT dependiendo del device y encuentra tiempos similares."""
    datos_csv = {}
    datos_txt = {}
    
    with open(archivo_csv, mode='r', newline='', encoding='utf-8') as csv_file:
        lector = csv.reader(csv_file)
        next(lector)  # Omitir encabezado
        for fila in lector:
            tiempo = datetime.strptime(fila[0], "%Y-%m-%d %H:%M:%S.%f")  # Modificado para aceptar milisegundos
            clave = (fila[1], tiempo)  # (device, tiempo)
            datos_csv[clave] = fila[2:]
    
    with open(archivo_txt, mode='r', encoding='utf-8') as txt_file:
        for linea in txt_file:
            fila = linea.strip().split(',')
            tiempo = datetime.strptime(fila[0], "%Y-%m-%d %H:%M:%S.%f")  # Modificado para aceptar milisegundos
            clave = (fila[1], tiempo)  # (device, tiempo)
            datos_txt[clave] = fila[2:]
    
    diferencias = []
    errores_totales = 0
    conteo_errores = 0
    
    for (device_csv, tiempo_csv), valores_csv in datos_csv.items():
        tiempos_similares = [(device_txt, tiempo_txt, valores_txt) for (device_txt, tiempo_txt), valores_txt in datos_txt.items()
                             if device_csv == device_txt and abs((tiempo_csv - tiempo_txt).total_seconds()) <= umbral_segundos]
        
        for _, tiempo_txt, valores_txt in tiempos_similares:
            if valores_csv != valores_txt:
                diferencias.append(f"Diferencia en device {device_csv}, tiempo {tiempo_csv} ~ {tiempo_txt}: CSV {valores_csv} vs TXT {valores_txt}")
                ax_csv = valores_csv[0]
                ay_csv = valores_csv[1]
                az_csv = valores_csv[2]
                (ax_text) = valores_txt[0]
                ay_text = valores_txt[1]
                az_text = valores_txt[2]
                error_ax = float(ax_csv) - float(ax_text)
                error_ay = float(ay_csv) - float(ay_text)
                error_az = float(az_csv) - float(az_text)
                promedios_x.append(error_ax)
                promedios_y.append(error_ay)                
                promedios_z.append(error_az)                
                # Calcular la diferencia en los valores ax, ay, az
                for v_csv, v_txt in zip(valores_csv, valores_txt):
                    try:
                        error = abs(float(v_csv) - float(v_txt))
                        errores_totales += error
                        conteo_errores += 1
                    except ValueError:
                        # Si no se puede convertir a float, continuar
                        continue
    
    
    if conteo_errores > 0:
        promedio_error = errores_totales / conteo_errores
        print(f"Promedio de error: {promedio_error:.4f}")
    else:
        print("No se encontraron diferencias en los valores numéricos.")
    
    if diferencias:
        print("Diferencias encontradas:")
        for dif in diferencias:
            print(dif)
    else:
        print("No hay diferencias significativas entre los archivos.")

# Ejemplo de uso
archivo_csv = 'datos.csv'  # Debe tener columnas: tiempo, device, ax, ay, az
archivo_txt = 'mentorinprueba1convertidos 2.txt'
convertir_csv_a_txt(archivo_csv, archivo_txt)
encontrar_tiempos_similares(archivo_csv, archivo_txt)
print("Promedios de X:", np.average(promedios_x))
print("Promedios de Y:", np.average(promedios_y))
print("Promedios de Z:", np.average(promedios_z))