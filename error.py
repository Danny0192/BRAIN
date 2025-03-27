import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Archivos ---
file_path = r'C:\Users\Danny\Desktop\Brain\BRAIN\quaterniorodilla.txt'
csv_path = r'C:\Users\Danny\Desktop\Brain\BRAIN\datosrodillader.csv'

# --- Leer archivo TXT del acelerómetro ---
df_acel = pd.read_csv(file_path, sep='\t')
df_acel.columns = ["Time", "ID", "Ax_cm/s2", "Ay_cm/s2", "Az_cm/s2", "Q0", "Q1", "Q2", "Q3"]

# Reemplazar comas por puntos
for col in ["Ax_cm/s2", "Ay_cm/s2", "Az_cm/s2", "Q0", "Q1", "Q2", "Q3"]:
    df_acel[col] = df_acel[col].astype(str).str.replace(',', '.').astype(float)

# --- Leer CSV de cámara ---
df_cam = pd.read_csv(csv_path, header=None)
df_cam.columns = ["Time", "ID", "Ax_cm/s2", "Ay_cm/s2", "Az_cm/s2"]
df_cam["ID"] = df_cam["ID"].astype(str)

# --- Transformar aceleración a coordenadas globales ---
def transformar_aceleracion_global(row):
    acc_local = np.array([row["Ax_cm/s2"], row["Ay_cm/s2"], row["Az_cm/s2"]])
    q = [row["Q0"], row["Q1"], row["Q2"], row["Q3"]]
    rot = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy usa orden [x, y, z, w]
    acc_global = rot.apply(acc_local)

    # CORREGIR LA GRAVEDAD EN Z
    acc_global[2] -= 981  # cm/s² (9.81 m/s²)

    return pd.Series(acc_global, index=["Ax_global", "Ay_global", "Az_global"])

df_acel[["Ax_global", "Ay_global", "Az_global"]] = df_acel.apply(transformar_aceleracion_global, axis=1)
df_acel["ID"] = df_acel["ID"].astype(str)

# --- Análisis de errores por ID ---
def merge_and_analyze_by_id(df_acel, df_cam):
    common_ids = set(df_acel['ID']) & set(df_cam['ID'])

    if not common_ids:
        print("No hay IDs en común entre los datasets.")
        return None

    results = []

    for current_id in common_ids:
        df_acel_id = df_acel[df_acel['ID'] == current_id]
        df_cam_id = df_cam[df_cam['ID'] == current_id]

        if df_acel_id.empty or df_cam_id.empty:
            continue

        # Error absoluto promedio por eje
        errors = {
            'ID': current_id,
            'Ax_error': abs(df_acel_id['Ax_global'].mean() - df_cam_id['Ax_cm/s2'].mean()),
            'Ay_error': abs(df_acel_id['Ay_global'].mean() - df_cam_id['Ay_cm/s2'].mean()),
            'Az_error': abs(df_acel_id['Az_global'].mean() - df_cam_id['Az_cm/s2'].mean())
        }
        results.append(errors)

    results_df = pd.DataFrame(results)

    print("\nResumen de Errores por ID:")
    print(results_df)

    # Error promedio por eje
    mean_err_x = results_df['Ax_error'].mean()
    mean_err_y = results_df['Ay_error'].mean()
    mean_err_z = results_df['Az_error'].mean()

    print("\nError Promedio Total:")
    print(f"  Eje X: {mean_err_x:.4f} cm/s²")
    print(f"  Eje Y: {mean_err_y:.4f} cm/s²")
    print(f"  Eje Z: {mean_err_z:.4f} cm/s²")

    # Calcular error porcentual con respecto al valor absoluto promedio
    mean_real_x = df_cam['Ax_cm/s2'].abs().mean()
    mean_real_y = df_cam['Ay_cm/s2'].abs().mean()
    mean_real_z = df_cam['Az_cm/s2'].abs().mean()

    err_pct_x = (mean_err_x / mean_real_x) * 100 if mean_real_x != 0 else 0
    err_pct_y = (mean_err_y / mean_real_y) * 100 if mean_real_y != 0 else 0
    err_pct_z = (mean_err_z / mean_real_z) * 100 if mean_real_z != 0 else 0

    print("\nError Porcentual Promedio:")
    print(f"  Eje X: {err_pct_x:.2f}%")
    print(f"  Eje Y: {err_pct_y:.2f}%")
    print(f"  Eje Z: {err_pct_z:.2f}%")

    return results_df

# --- Diagnóstico general ---
print("Información general:")
print(f"IDs en Acelerómetro: {sorted(df_acel['ID'].unique())}")
print(f"IDs en Cámara: {sorted(df_cam['ID'].unique())}")
print(f"Registros en Acelerómetro: {len(df_acel)}")
print(f"Registros en Cámara: {len(df_cam)}")

# --- Ejecutar análisis ---
merge_and_analyze_by_id(df_acel, df_cam)