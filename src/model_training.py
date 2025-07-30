import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import xgboost as xgb


def train_random_forest(X_train, y_train):
    print("=== MODEL TRAINING: RandomForestRegressor ===")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("=== MODEL EVALUATION FUNCTION ===")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R2: {r2:.4f}")
    return y_pred, rmse, mae, r2

def custom_score(y_true, y_pred):
    # Example: penalize late predictions more (as in some RUL papers)
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1)
    return np.mean(score)

def tune_random_forest(X_train, y_train, groups=None, n_iter=40, cv=5, random_state=42):
    print("=== HYPERPARAMETER TUNING: RandomizedSearchCV ===")
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500, 700, 1000],
        'max_depth': [None, 10, 20, 30, 40, 50, 70, 100],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
        'bootstrap': [True, False]
    }
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    if groups is not None:
        gkf = GroupKFold(n_splits=cv)
        cv_split = gkf.split(X_train, y_train, groups=groups)
    else:
        cv_split = cv
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_split,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best RMSE (CV):", -search.best_score_)
    return search.best_estimator_

def train_xgboost(X_train, y_train):
    print("=== MODEL TRAINING: XGBoost ===")
    model = xgb.XGBRegressor(
        n_estimators=45,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.99,
        random_state=42,
        n_jobs=-1,
        reg_lambda=4,
        reg_alpha=0.1,
    )
    model.fit(X_train, y_train)
    return model

def tune_xgboost(X_train, y_train, groups=None, n_iter=40, cv=5, random_state=42):
    print("=== HYPERPARAMETER TUNING: XGBoost (RandomizedSearchCV, GroupKFold) ===")
    param_dist = {
        'n_estimators': [40, 50, 60, 80, 100],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.005, 0.01, 0.015, 0.02],
        'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.05, 0.1, 0.2],
        'reg_alpha': [0.05, 0.1, 0.2, 0.5],
        'reg_lambda': [2, 3, 4, 5]
    }
    model = xgb.XGBRegressor(random_state=random_state, n_jobs=-1, tree_method='hist')
    if groups is not None:
        gkf = GroupKFold(n_splits=cv)
        cv_split = gkf.split(X_train, y_train, groups=groups)
    else:
        cv_split = cv
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_split,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best RMSE (CV):", -search.best_score_)
    return search.best_estimator_