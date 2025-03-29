import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Leer archivo
file_path = r'C:\Users\Danny\Desktop\Brain\BRAIN\RODILLAS2052quaternos.txt'
df = pd.read_csv(file_path, sep='\t')
df.columns = ["Time", "ID", "Ax", "Ay", "Az", "Q0", "Q1", "Q2", "Q3"]

# Reemplazar comas por puntos
for col in ["Ax", "Ay", "Az", "Q0", "Q1", "Q2", "Q3"]:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

df["ID"] = df["ID"].astype(str)

# Gravedad en sistema global (en dirección Y hacia arriba)
g_vector_global = np.array([0, 981, 0])  # en cm/s²

# Función para transformar aceleración local a global y corregir gravedad proyectada
def transformar_y_corregir(row):
    acc_local = np.array([row["Ax"], row["Ay"], row["Az"]])
    quat = [row["Q0"], row["Q1"], row["Q2"], row["Q3"]]  # w, x, y, z
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: x, y, z, w

    acc_global = rot.apply(acc_local)

    # Rotar gravedad desde global al sistema del sensor
    gravity_sensor = rot.inv().apply(g_vector_global)

    # Transformar nuevamente la gravedad al sistema global para cancelarla
    gravity_global_proj = rot.apply(gravity_sensor)

    # Corregir aceleración eliminando componente gravitacional
    acc_corrected = acc_global - gravity_global_proj

    return pd.Series(acc_corrected, index=["Ax_global", "Ay_global", "Az_global"])

# Aplicar transformación
df[["Ax_global", "Ay_global", "Az_global"]] = df.apply(transformar_y_corregir, axis=1)

# Guardar y mostrar resultados
df[["Time", "ID", "Ax_global", "Ay_global", "Az_global"]].to_csv("aceleraciones_corregidas.txt", sep='\t', index=False)
print("\nPrimeras filas corregidas:")
print(df[["Time", "ID", "Ax_global", "Ay_global", "Az_global"]].head(10))
