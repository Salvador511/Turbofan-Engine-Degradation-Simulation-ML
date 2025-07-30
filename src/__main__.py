from src import (
    load_cmaps_data, add_rul_labels, normalize_by_unit, plot_sensor_trends,
    train_random_forest, evaluate_model, custom_score, tune_random_forest,
    train_xgboost, tune_xgboost
)
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

    df_train = df_train.drop(columns=['unit', 'time'])
    df_test = df_test.drop(columns=['unit', 'time'])

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

    print("\n" + "="*60 + "\n")

    # === MODEL SELECTION & TRAINING ===
    print("=== MODEL SELECTION & TRAINING ===")

    # --- RANDOM FOREST ---
    # rf_model = train_random_forest(X_train, y_train)
    # (Opcional) Tuning:
    # rf_model = tune_random_forest(X_train, y_train)

    # --- XGBOOST ---
    xgb_model = train_xgboost(X_train, y_train)
    # (Opcional) Tuning:
    #xgb_model = tune_xgboost(X_train, y_train)

    print("\n" + "="*60 + "\n")

    print("=== MODEL EVALUATION ===")
    # y_pred, rmse, mae, r2 = evaluate_model(rf_model, X_test, y_test)  # Random Forest
    y_pred, rmse, mae, r2 = evaluate_model(xgb_model, X_test, y_test)   # XGBoost
    print(f"Custom Score: {custom_score(y_test, y_pred):.2f}")

if __name__ == "__main__":
    main()