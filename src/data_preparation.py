import pandas as pd
import numpy as np

def load_cmaps_data(filepath):
    """
    Carga un archivo C-MAPSS (por ejemplo, train_FD001.txt) y devuelve un DataFrame limpio.
    - Elimina columnas vacías al final.
    - Asigna nombres de columna estándar (unit, time, op_setting_1-3, sensor_1-21).
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    # Eliminar columnas completamente vacías (pueden aparecer por espacios extra)
    df = df.dropna(axis=1, how='all')
    # Asignar nombres de columna (solo hasta sensor_21)
    columns = (
        ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )
    df = df.iloc[:, :26]  # Solo las primeras 26 columnas (por si hay sensores extra)
    df.columns = columns
    return df

def add_rul_labels(df, max_rul=200):
    """
    Añade una columna 'RUL' al DataFrame.
    RUL = ciclo máximo de la unidad - ciclo actual, limitado por max_rul.
    """
    max_cycles = df.groupby('unit')['time'].transform('max')
    df['RUL'] = (max_cycles - df['time']).clip(upper=max_rul)
    return df

def normalize_by_unit(df):
    """
    Normaliza los sensores sensor_1 a sensor_21 por unidad usando Min-Max normalization.
    No usa sklearn, solo pandas.
    """
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    def min_max_norm(group):
        return (group[sensor_cols] - group[sensor_cols].min()) / (group[sensor_cols].max() - group[sensor_cols].min())
    normed = df.groupby('unit', group_keys=False).apply(
        lambda g: g.assign(**min_max_norm(g))
    )
    df[sensor_cols] = normed[sensor_cols]
    return df

def drop_useless_sensors(df):
    """
    Elimina sensores que no aportan información relevante.
    """
    USELESS_SENSORS = ['op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    return df.drop(columns=USELESS_SENSORS, errors='ignore')

# def add_advanced_features(df, windows=[5, 10, 20]):
#     """
#     Agrega rolling mean, std, min, max, diff y tendencia para cada sensor por unidad.
#     """
#     sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
#     for window in windows:
#         for col in sensor_cols:
#             df[f'{col}_mean_{window}'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
#             df[f'{col}_std_{window}'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std())
#             df[f'{col}_min_{window}'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).min())
#             df[f'{col}_max_{window}'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).max())
#             df[f'{col}_diff_{window}'] = df.groupby('unit')[col].transform(lambda x: x.diff(window).fillna(0))
#             # Tendencia (pendiente de regresión lineal en la ventana)
#             df[f'{col}_trend_{window}'] = df.groupby('unit')[col].transform(
#                 lambda x: x.rolling(window, min_periods=1).apply(
#                     lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=False
#                 )
#             )
#     return df

def add_advanced_features(df, windows=[5, 10, 20]):
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    new_features = []
    for window in windows:
        for col in sensor_cols:
            new_features.append(
                df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean()).rename(f'{col}_mean_{window}')
            )
    df_new = pd.concat([df] + new_features, axis=1)
    return df_new.copy()

# ...end of file...