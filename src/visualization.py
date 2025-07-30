"""
Módulo de visualización para el dataset CMAPSS.
Crea gráficos interactivos y los guarda como .html en /outputs/plots.
"""
import plotly.express as px
import plotly.graph_objects as go
import os

def plot_sensor_trends(df, output_dir, unit_id=1):
    """
    Gráfico de líneas de la evolución de sensores para un motor específico.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        output_dir (str): Carpeta donde guardar el gráfico
        unit_id (int): ID del motor a graficar
    """
    df_unit = df[df['unit'] == unit_id]
    sensors = [col for col in df.columns if col.startswith('s')]
    fig = px.line(df_unit, x='cycle', y=sensors, title=f'Evolución de sensores - Motor {unit_id}')
    fig.write_html(os.path.join(output_dir, f'sensor_trends_unit{unit_id}.html'))

def plot_rul_distribution(df, output_dir):
    """
    Histograma de la distribución de RUL.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        output_dir (str): Carpeta donde guardar el gráfico
    """
    fig = px.histogram(df, x='RUL', nbins=50, title='Distribución de RUL')
    fig.write_html(os.path.join(output_dir, 'rul_distribution.html'))

def plot_correlation_heatmap(df, output_dir):
    """
    Mapa de calor de correlación entre sensores útiles y RUL.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        output_dir (str): Carpeta donde guardar el gráfico
    """
    sensors = [col for col in df.columns if col.startswith('s')]
    corr = df[sensors + ['RUL']].corr()
    fig = px.imshow(corr, text_auto=True, title='Correlación entre sensores y RUL')
    fig.write_html(os.path.join(output_dir, 'correlation_heatmap.html'))

def plot_sensor_boxplots(df, output_dir):
    """
    Boxplots de sensores útiles.
    Args:
        df (pd.DataFrame): DataFrame de entrada
        output_dir (str): Carpeta donde guardar el gráfico
    """
    sensors = [col for col in df.columns if col.startswith('s')]
    for sensor in sensors:
        fig = px.box(df, y=sensor, title=f'Boxplot {sensor}')
        fig.write_html(os.path.join(output_dir, f'boxplot_{sensor}.html'))
