from src import load_cmaps_data, add_rul_labels, normalize_by_unit, plot_sensor_trends
import numpy as np

def main():
    print("=== DATA PREPARATION: TRAIN ===")
    train_path = "./CMAPSSData/train_FD001.txt"
    df_train = load_cmaps_data(train_path)
    print("  - Data loaded")
    df_train = add_rul_labels(df_train)
    print("  - RUL calculated")
    df_train = normalize_by_unit(df_train)
    print("  - Sensors normalized")
    print("Train shape:", df_train.shape)
    print("Train sample:")
    print(df_train.head())

    print("\n" + "="*60 + "\n")

    print("=== DATA PREPARATION: TEST ===")
    test_path = "./CMAPSSData/test_FD001.txt"
    rul_path = "./CMAPSSData/RUL_FD001.txt"
    df_test = load_cmaps_data(test_path)
    print("  - Data loaded")
    df_test = normalize_by_unit(df_test)
    print("  - Sensors normalized")
    rul_true = np.loadtxt(rul_path)
    max_cycles = df_test.groupby('unit')['time'].max().values
    df_test['RUL'] = 0
    for idx, (unit, rul_offset) in enumerate(zip(sorted(df_test['unit'].unique()), rul_true)):
        mask = df_test['unit'] == unit
        df_test.loc[mask, 'RUL'] = (max_cycles[idx] - df_test.loc[mask, 'time']) + rul_offset

    last_cycles = df_test.groupby('unit').tail(1)
    print("Test shape:", df_test.shape)
    print("Test sample (last cycles):")
    print(last_cycles.head())

    print("\n" + "="*60 + "\n")

    print("=== VISUALIZATION ===")
    plot_sensor_trends(df_train, unit_id=1, sensors=['sensor_2', 'sensor_3'], output_path="sensor_trends_unit1.html")
    print("Plot saved to sensor_trends_unit1.html")

    print("\n" + "="*60 + "\n")

    print("=== READY FOR MACHINE LEARNING ===")
    feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    X_train = df_train[feature_cols]
    y_train = df_train['RUL']
    X_test = last_cycles[feature_cols]
    y_test = rul_true

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

if __name__ == "__main__":
    main()