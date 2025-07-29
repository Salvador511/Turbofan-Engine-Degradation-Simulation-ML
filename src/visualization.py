import plotly.graph_objs as go

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
# ...end of file...