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
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    model = build_mlp(X_train.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    # Calculate R2
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    # Save results
    with open(f'{output_dir}/mlp_results.txt', 'w') as f:
        f.write(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}\n')
    return {'rmse': rmse, 'mae': mae, 'r2': r2}
