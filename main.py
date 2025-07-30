
"""
Main script to run the predictive maintenance pipeline with CMAPSS.
"""
import os
from src.data_preparation import prepare_data
from src.visualization import plot_sensor_trends, plot_rul_distribution, plot_correlation_heatmap, plot_sensor_boxplots
from src.model_training import train_and_evaluate

DATA_PATH = 'CMAPSSData/train_FD001.txt'
OUTPUT_PLOTS = 'outputs/plots'
OUTPUTS = 'outputs'

def main():
    # Data preparation
    print('Preparing data...')
    df = prepare_data(DATA_PATH)

    # Visualization
    print('Generating visualizations...')
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)
    plot_sensor_trends(df, OUTPUT_PLOTS, unit_id=1)
    plot_rul_distribution(df, OUTPUT_PLOTS)
    plot_correlation_heatmap(df, OUTPUT_PLOTS)
    plot_sensor_boxplots(df, OUTPUT_PLOTS)

    # Model training and evaluation
    print('Training and evaluating model...')
    os.makedirs(OUTPUTS, exist_ok=True)
    results = train_and_evaluate(df, OUTPUTS)
    print(f"Results - RMSE: {results['rmse']:.2f}, MAE: {results['mae']:.2f}, R2: {results['r2']:.4f}")

if __name__ == '__main__':
    main()
