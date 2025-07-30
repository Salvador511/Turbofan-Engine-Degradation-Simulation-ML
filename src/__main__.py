from src import (
    load_cmaps_data, add_rul_labels, normalize_by_unit, plot_sensor_trends,
    evaluate_model, custom_score, tune_xgboost, train_xgboost
)
from src.data_preparation import drop_useless_sensors, add_advanced_features
from src.visualization import plot_correlation_heatmap, plot_rul_distribution, plot_rul_trends
from src.model_training import train_xgboost, train_random_forest
import numpy as np
import pandas as pd

def main():
    print("=== DATA PREPARATION: TRAIN ===")
    train_path = "./CMAPSSData/train_FD001.txt"
    df_train = load_cmaps_data(train_path)
    print("  - Data loaded")
    df_train = add_rul_labels(df_train)
    print("  - RUL calculated")
    df_train = normalize_by_unit(df_train)
    print("  - Sensors normalized")
    df_train = drop_useless_sensors(df_train)
    print("  - Useless sensors dropped")
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
    df_test = drop_useless_sensors(df_test)
    print("  - Useless sensors dropped")

    # Add RUL to test data
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
    plot_rul_distribution(df_train, output_path="rul_distribution.png")
    print("RUL distribution plot saved to rul_distribution.png")
    plot_rul_trends(df_train, units=[1, 2, 3, 4, 5], output_path="rul_trends.html")
    print("RUL trends plot saved to rul_trends.html")
    plot_correlation_heatmap(df_train, output_path="correlation_heatmap.png")
    print("Correlation heatmap saved to correlation_heatmap.png")

    print("\n" + "="*60 + "\n")

    # Eliminar columnas no necesarias para el modelo
    df_train = df_train.drop(columns=[ 'time'])
    df_test = df_test.drop(columns=[ 'time'])

    # Sensores originales más importantes (ajusta según tu importancia de features)
    top_sensors = ['sensor_11', 'sensor_12', 'sensor_4', 'sensor_7', 'sensor_15', 'sensor_9']

    for window in [5, 10, 20]:
        for col in top_sensors:
            trend_col = f"{col}_trend_{window}"
            df_train[trend_col] = df_train.groupby('unit')[col].transform(
                lambda x: x.rolling(window, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=False
                )
            )
            df_test[trend_col] = df_test.groupby('unit')[col].transform(
                lambda x: x.rolling(window, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=False
                )
            )

    df_train = df_train.drop(columns=['unit'])
    df_test = df_test.drop(columns=['unit'])

    print("=== READY FOR MACHINE LEARNING ===")
    # Incluye los trend en la selección de features
    feature_cols = [col for col in df_train.columns if col.startswith('sensor_') or '_trend_' in col]
    correlations = df_train[feature_cols + ['RUL']].corr()['RUL'].abs().sort_values(ascending=False)
    top_features = correlations.index[1:21]  # Prueba con 10, 20, 40
    feature_cols = [col for col in feature_cols if col in top_features]

    feature_cols = [col for col in feature_cols if col in last_cycles.columns]

    X_train = df_train[feature_cols]
    y_train = df_train['RUL']
    X_test = last_cycles[feature_cols]
    y_test = last_cycles['RUL']

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\n" + "="*60 + "\n")

    print("=== MODEL SELECTION & TRAINING ===")
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    print("\n--- XGBoost Feature Importances ---")
    importances_xgb = xgb_model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances_xgb), key=lambda x: -x[1])[:20]:
        print(f"{col}: {imp:.4f}")
    print("Features usados en el modelo:", feature_cols)
    print("\n=== XGBoost EVALUATION ===")
    y_pred_xgb, rmse_xgb, mae_xgb, r2_xgb = evaluate_model(xgb_model, X_test, y_test)
    print(f"Custom Score (XGBoost): {custom_score(y_test, y_pred_xgb):.2f}")

    # RandomForest
    # rf_model = train_random_forest(X_train, y_train)
    # print("\n--- RandomForest Feature Importances ---")
    # importances_rf = rf_model.feature_importances_
    # for col, imp in sorted(zip(feature_cols, importances_rf), key=lambda x: -x[1])[:20]:
    #     print(f"{col}: {imp:.4f}")

    # print("\n=== RandomForest EVALUATION ===")
    # y_pred_rf, rmse_rf, mae_rf, r2_rf = evaluate_model(rf_model, X_test, y_test)
    # print(f"Custom Score (RandomForest): {custom_score(y_test, y_pred_rf):.2f}")

if __name__ == "__main__":
    main()