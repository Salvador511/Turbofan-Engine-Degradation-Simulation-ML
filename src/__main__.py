from src import load_cmaps_data, add_rul_labels, normalize_by_unit, plot_sensor_trends

def main():
    data_path = "./CMAPSSData/train_FD001.txt"
    plot_path = "sensor_trends_unit1.html"
    df = load_cmaps_data(data_path)
    df = add_rul_labels(df)
    df = normalize_by_unit(df)
    plot_sensor_trends(df, unit_id=1, sensors=['sensor_2', 'sensor_3'], output_path=plot_path)
    print(f"Pipeline ejecutado. Gráfico guardado en {plot_path}")

if __name__ == "__main__":
    main()
# ...end of file...