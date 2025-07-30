"""
Module for training and evaluating models for RUL prediction.
Trains an MLP and evaluates with RMSE, MAE, and R2.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def prepare_train_test(df, test_size=0.2, random_state=42):
    """
    Prepare training and test sets.
    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Test set proportion
        random_state (int): Random seed
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(['unit', 'cycle', 'RUL'], axis=1).values
    y = df['RUL'].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_mlp(input_dim):
    """
    Build a simple MLP model for regression.
    Args:
        input_dim (int): Input dimension
    Returns:
        tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate(df, output_dir):
    """
    Train the model and evaluate with RMSE, MAE, and R2.
    Args:
        df (pd.DataFrame): Input DataFrame
        output_dir (str): Directory to save results
    Returns:
        dict: Evaluation results
    """
    import pandas as pd
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    # Hyperparameters
    epochs = 20
    batch_size = 64
    val_split = 0.1
    model = build_mlp(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=0)
    # Save the trained model
    model.save(f'{output_dir}/mlp_model.h5')
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    # Save results to txt
    with open(f'{output_dir}/mlp_results.txt', 'w') as f:
        f.write(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}\n')
    # Save results to CSV (metrics and hyperparameters)
    results_dict = {
        'epochs': [epochs],
        'batch_size': [batch_size],
        'validation_split': [val_split],
        'rmse': [rmse],
        'mae': [mae],
        'r2': [r2]
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'{output_dir}/mlp_results.csv', index=False)
    # Save training history (loss and val_loss per epoch)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f'{output_dir}/mlp_training_history.csv', index=False)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}
