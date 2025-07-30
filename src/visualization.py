import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_sensor_trends(df, unit_id=1, sensors=['sensor_2', 'sensor_3'], output_path='plot.html'):
    """
    Visualiza la evolución temporal de uno o varios sensores para una unidad específica.
    Guarda el gráfico como HTML en la ruta indicada.
    """
    unit_df = df[df['unit'] == unit_id]
    traces = []
    for sensor in sensors:
        traces.append(go.Scatter(
            x=unit_df['time'],
            y=unit_df[sensor],
            mode='lines',
            name=sensor
        ))
    layout = go.Layout(
        title=f'Sensor trends for unit {unit_id}',
        xaxis=dict(title='Cycle'),
        yaxis=dict(title='Normalized Sensor Measurement'),
        legend=dict(title='Sensors')
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(output_path)

def plot_correlation_heatmap(df, output_path='correlation_heatmap.png'):
    """
    Genera un heatmap de correlación entre sensores y la RUL.
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    corr = df[sensor_cols + ['RUL']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap (Sensores y RUL)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rul_distribution(df, output_path='rul_distribution.png'):
    """
    Visualiza la distribución del RUL en el dataset.
    También muestra un boxplot para identificar outliers.
    """
    plt.figure(figsize=(12, 8))
    
    # Crear un subplot con 2 filas y 1 columna
    plt.subplot(2, 1, 1)
    sns.histplot(df['RUL'], bins=50, kde=True)
    plt.title('Distribución de RUL (Remaining Useful Life)')
    plt.xlabel('RUL (ciclos)')
    plt.ylabel('Frecuencia')
    
    # Añadir boxplot para mostrar outliers
    plt.subplot(2, 1, 2)
    sns.boxplot(x=df['RUL'])
    plt.title('Boxplot de RUL')
    plt.xlabel('RUL (ciclos)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rul_trends(df, units=None, output_path='rul_trends.html'):
    """
    Visualiza la evolución del RUL a lo largo de los ciclos para unidades específicas.
    Si no se especifican unidades, se seleccionan aleatoriamente 5.
    """
    if units is None:
        # Seleccionar aleatoriamente 5 unidades para visualizar
        all_units = df['unit'].unique()
        units = np.random.choice(all_units, size=min(5, len(all_units)), replace=False)
    
    traces = []
    for unit in units:
        unit_df = df[df['unit'] == unit].sort_values('time')
        traces.append(go.Scatter(
            x=unit_df['time'],
            y=unit_df['RUL'],
            mode='lines',
            name=f'Unit {unit}'
        ))
    
    layout = go.Layout(
        title='Evolución del RUL por Ciclo',
        xaxis=dict(title='Ciclo'),
        yaxis=dict(title='RUL (ciclos restantes)'),
        legend=dict(title='Unidades')
    )
    
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(output_path)