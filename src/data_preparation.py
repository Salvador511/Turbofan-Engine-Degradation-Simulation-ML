import pandas as pd

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

def add_rul_labels(df, max_rul=125):
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
# ...end of file...