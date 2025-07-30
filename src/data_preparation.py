"""
Módulo de preparación de datos para el dataset CMAPSS.
Carga, calcula RUL, normaliza y selecciona sensores útiles.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_cmapss_data(filepath):
    """
    Carga el archivo de datos CMAPSS.
    Args:
        filepath (str): Ruta al archivo train_FD001.txt
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    # El dataset no tiene cabecera, columnas según documentación
    columns = [
        'unit', 'cycle',
        'op_setting_1', 'op_setting_2', 'op_setting_3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
    ]
    df = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    return df

def calculate_rul(df):
    """
    Calcula el Remaining Useful Life (RUL) para cada ciclo de cada motor.
    Args:
        df (pd.DataFrame): DataFrame con los datos originales
    Returns:
        pd.DataFrame: DataFrame con columna 'RUL' añadida
    """
    rul_df = df.groupby('unit')['cycle'].max().reset_index()
    rul_df.columns = ['unit', 'max_cycle']
    df = df.merge(rul_df, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def normalize_features(df, feature_cols):
    """
    Normaliza las columnas numéricas por motor (unit) usando MinMaxScaler.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        feature_cols (list): Lista de columnas a normalizar
    Returns:
        pd.DataFrame: DataFrame con columnas normalizadas
    """
    df_norm = df.copy()
    scaler = MinMaxScaler()
    for unit in df['unit'].unique():
        idx = df['unit'] == unit
        df_norm.loc[idx, feature_cols] = scaler.fit_transform(df.loc[idx, feature_cols])
    return df_norm

def select_useful_sensors(df):
    """
    Selecciona las columnas de sensores útiles según el artículo base.
    Args:
        df (pd.DataFrame): DataFrame de entrada
    Returns:
        pd.DataFrame: DataFrame solo con sensores útiles
    """
    # Según el artículo, sensores útiles para FD001: s2, s3, s4, s7, s8, s9, s11, s12, s13, s15, s17, s20, s21
    useful_sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's15', 's17', 's20', 's21']
    cols = ['unit', 'cycle'] + useful_sensors + ['RUL']
    return df[cols]

def prepare_data(filepath):
    """
    Pipeline de preparación de datos: carga, RUL, normalización y selección de sensores útiles.
    Args:
        filepath (str): Ruta al archivo train_FD001.txt
    Returns:
        pd.DataFrame: DataFrame listo para modelado
    """
    df = load_cmapss_data(filepath)
    df = calculate_rul(df)
    feature_cols = [col for col in df.columns if col.startswith('s')]
    df = normalize_features(df, feature_cols)
    df = select_useful_sensors(df)
    return df
