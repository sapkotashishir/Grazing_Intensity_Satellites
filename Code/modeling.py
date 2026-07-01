
#%%
# Install dependencies



#%%
# Imports
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import optuna
import xgboost as xgb
# import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, ParameterSampler, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42

#%%
# Configuration
DATA_PATH = Path('/home/shishir/Documents/Graze_Sat/Results/grazing_satellite_features_grid10m.csv')
OUTPUT_DIR = Path('/home/shishir/Documents/Graze_Sat/Results/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = 'au_days'
GROUP_COL = 'cell_id'
TIME_COL = 'bin_start'
BIN_COL = 'bin_idx'
KEEP_BINS = [0, 1, 2, 3]
TEST_SIZE = 0.30
N_SPLITS_CV = 5
N_ITER_TREE_SEARCH = 10
N_ITER_SVM_SEARCH = 10
N_TRIALS_MLP = 10

#%%
# Helper functions

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f'Loaded data: {df.shape[0]} rows x {df.shape[1]} columns')
    return df


def filter_active_grazing_bins(df: pd.DataFrame, keep_bins=None) -> pd.DataFrame:
    keep_bins = KEEP_BINS if keep_bins is None else keep_bins
    filtered = df[df[BIN_COL].isin(keep_bins)].copy()
    print(f'Filtered to active bins {keep_bins}: {filtered.shape[0]} rows')
    return filtered


def drop_first_bin_change_nans(df: pd.DataFrame) -> pd.DataFrame:
    change_cols = [c for c in df.columns if c.endswith('_change')]
    cleaned = df.dropna(subset=change_cols, how='all').copy()
    print(f'Rows after dropping all-NaN change rows: {cleaned.shape[0]}')
    return cleaned


def remove_iqr_outliers(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()
    print(f'Outlier removal on {target_col}: removed {len(df) - len(filtered)} rows')
    return filtered


# def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
#     out = df.copy()
#     out['bin_month'] = pd.to_datetime(out[TIME_COL]).dt.month
#     return out


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude_cols = {GROUP_COL, TIME_COL, TARGET_COL,'s2_count',
       's1_count', 's2_interpolated', 's1_interpolated','n_animals',
       'animal_days_records', 'total_animal_hours',
                     'total_forage_removal'}
    
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and df[c].dtype.kind in 'fi'
    ]
    return feature_cols


def prepare_ml_data(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y_raw = df[TARGET_COL].values
    y_log = np.log1p(y_raw)
    groups = df[GROUP_COL].values
    return X, y_raw, y_log, groups, feature_cols


def group_train_test_split(X, y_log, y_raw, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y_log, groups=groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), 'Group leakage detected.'
    split = {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'X_train': X[train_idx],
        'X_test': X[test_idx],
        'y_train_log': y_log[train_idx],
        'y_test_log': y_log[test_idx],
        'y_train_raw': y_raw[train_idx],
        'y_test_raw': y_raw[test_idx],
        'groups_train': groups[train_idx],
        'groups_test': groups[test_idx],
    }
    print(f"Train rows: {len(train_idx)} | unique cells: {len(set(groups[train_idx]))}")
    print(f"Test rows: {len(test_idx)} | unique cells: {len(set(groups[test_idx]))}")
    return split


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }


def predict_raw(model, X):
    log_preds = model.predict(X)
    return np.clip(np.expm1(log_preds), a_min=0, a_max=None)


def tune_model(model_cls, param_dist, fixed_params, X_train, y_train, cv_splits, n_iter=50, desc='Model'):
    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE)
    best_score = float('inf')
    best_params = None
    for params in tqdm(sampler, desc=desc):
        model = model_cls(**params, **fixed_params)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv_splits,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
        )
        avg_mse = -np.mean(scores)
        if avg_mse < best_score:
            best_score = avg_mse
            best_params = params
    best_model = model_cls(**best_params, **fixed_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params, best_score


def build_mlp_objective(X_train, y_train, cv_splits):
    def objective(trial):
        params = {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes',
                [(50,), (100,), (50, 50), (100, 50), (100, 100)]
            ),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'max_iter': trial.suggest_categorical('max_iter', [200, 300, 500]),
            'random_state': RANDOM_STATE,
            'early_stopping': True,
            'n_iter_no_change': 20,
            'tol': 1e-4,
        }
        fold_mses = []
        for train_fold_idx, valid_fold_idx in cv_splits:
            X_train_fold = X_train[train_fold_idx]
            X_valid_fold = X_train[valid_fold_idx]
            y_train_fold = y_train[train_fold_idx]
            y_valid_fold = y_train[valid_fold_idx]
            scaler = StandardScaler()
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_valid_fold_scaled = scaler.transform(X_valid_fold)
            model = MLPRegressor(**params)
            model.fit(X_train_fold_scaled, y_train_fold)
            preds = model.predict(X_valid_fold_scaled)
            fold_mses.append(mean_squared_error(y_valid_fold, preds))
        return float(np.mean(fold_mses))
    return objective

#%%
# Load and clean data
analysis_df = load_data(DATA_PATH)
analysis_df = (
    analysis_df
    .dropna()
    .reset_index(drop=True)
)
print('Target summary after cleaning:')
print(analysis_df[TARGET_COL].describe())



#%%
# EDA plots
feature_cols_eda = [
    c for c in analysis_df.columns
    if c not in ['cell_id', 'bin_idx', 'bin_start', 'au_days', 'total_forage_removal', 'n_animals','animal_days_records', 'total_animal_hours','n_animals','s1_count','s2_count','s1_interpolated','s2_interpolated']
    and not c.endswith('_change')
]
change_cols_eda = [f'{c}_change' for c in feature_cols_eda if f'{c}_change' in analysis_df.columns]
core_cols = [TARGET_COL] + feature_cols_eda + change_cols_eda
core_cols = [c for c in core_cols if c in analysis_df.columns]
numeric_core = analysis_df[core_cols].select_dtypes(include=[np.number])

plt.figure(figsize=(max(12, 0.6 * len(numeric_core.columns)), 8))
sns.heatmap(numeric_core.corr(), annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(analysis_df[TARGET_COL], bins=30, kde=True, color='green')
plt.title('Distribution of Animal Unit Days')
plt.xlabel('Animal Unit Days')
plt.ylabel('Frequency Count')
plt.tight_layout()
plt.show()

#%%
# Prepare machine learning data
X, y_raw, y_log, groups, feature_cols = prepare_ml_data(analysis_df)
split = group_train_test_split(X, y_log, y_raw, groups)

X_train = split['X_train']
X_test = split['X_test']
y_train_log = split['y_train_log']
y_test_log = split['y_test_log']
y_train_raw = split['y_train_raw']
y_test_raw = split['y_test_raw']
groups_train = split['groups_train']

cv_splits = list(GroupKFold(n_splits=N_SPLITS_CV).split(X_train, y_train_log, groups=groups_train))

scaler_global = StandardScaler()
X_train_scaled = scaler_global.fit_transform(X_train)
X_test_scaled = scaler_global.transform(X_test)

print(f'Number of features used: {len(feature_cols)}')
print(feature_cols)

#%%
# Hyperparameter spaces
rf_param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

xgb_param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1, 5],
    'reg_lambda': [1, 5, 10],
}

# SVR is scale-sensitive, so it is always tuned/fit on X_train_scaled / X_test_scaled
# (built in the "Prepare machine learning data" cell above), never on raw X_train.
svm_param_dist = {
    'C': [0.1, 1, 10, 50, 100],
    'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto'],
}

#%%
# Tune tree-based models
print('Tuning Random Forest...')
rf_model, rf_params, rf_cv_mse = tune_model(
    RandomForestRegressor,
    rf_param_dist,
    {'random_state': RANDOM_STATE, 'n_jobs': -1},
    X_train,
    y_train_log,
    cv_splits,
    n_iter=N_ITER_TREE_SEARCH,
    desc='RF',
)

print('Tuning XGBoost...')
xgb_model, xgb_params, xgb_cv_mse = tune_model(
    xgb.XGBRegressor,
    xgb_param_dist,
    {
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',
    },
    X_train,
    y_train_log,
    cv_splits,
    n_iter=N_ITER_TREE_SEARCH,
    desc='XGBoost',
)

print('Best CV MSE values:')
print(f'RF: {rf_cv_mse:.4f} | params: {rf_params}')
print(f'XGBoost: {xgb_cv_mse:.4f} | params: {xgb_params}')

#%%
# Tune SVM (on scaled features)
print('Tuning SVM...')
svm_model, svm_params, svm_cv_mse = tune_model(
    SVR,
    svm_param_dist,
    {},
    X_train_scaled,
    y_train_log,
    cv_splits,
    n_iter=N_ITER_SVM_SEARCH,
    desc='SVM',
)

print(f'SVM: {svm_cv_mse:.4f} | params: {svm_params}')

#%%
# Tune MLP with Optuna
print('Tuning MLP with Optuna...')
mlp_study = optuna.create_study(direction='minimize', study_name='mlp_tuning')
mlp_study.optimize(
    build_mlp_objective(X_train, y_train_log, cv_splits),
    n_trials=N_TRIALS_MLP,
    n_jobs=-1,
    show_progress_bar=True,
)

print(f'Best MLP CV MSE: {mlp_study.best_value:.4f}')
print(f'Best MLP params: {mlp_study.best_params}')

mlp_model = MLPRegressor(
    **mlp_study.best_params,
    early_stopping=True,
    n_iter_no_change=20,
    tol=1e-4,
    random_state=RANDOM_STATE,
)
mlp_model.fit(X_train_scaled, y_train_log)

#%%
# Compare models and ensemble
rf_preds = predict_raw(rf_model, X_test)
xgb_preds = predict_raw(xgb_model, X_test)
svm_preds = predict_raw(svm_model, X_test_scaled)
mlp_preds = predict_raw(mlp_model, X_test_scaled)
ensemble_preds = (rf_preds + xgb_preds + svm_preds + mlp_preds) / 4.0

predictions = {
    'Random Forest': rf_preds,
    'XGBoost': xgb_preds,
    'SVM': svm_preds,
    'MLP': mlp_preds,
    'Ensemble (RF+XGB+SVM+MLP)': ensemble_preds,
}

results = []
for name, preds in predictions.items():
    metrics = evaluate_predictions(y_test_raw, preds)
    results.append({
        'model': name,
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2'],
    })

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df.to_csv(OUTPUT_DIR / 'model_results_10m.csv', index=False)
results_df

#%%
# Champion model and prediction export
champion_name = results_df.iloc[0]['model']
champion_preds = predictions[champion_name]
print(f'Champion model: {champion_name}')

predictions_df = pd.DataFrame({
    'actual_au_days': y_test_raw,
    'pred_rf': rf_preds,
    'pred_xgb': xgb_preds,
    'pred_svm': svm_preds,
    'pred_mlp': mlp_preds,
    'pred_ensemble': ensemble_preds,
})
predictions_df.to_csv(OUTPUT_DIR / 'test_predictions_10m.csv', index=False)
predictions_df.head()

#%%
# Feature importance plot (tree-based importance, XGBoost)
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(15))

plt.figure(figsize=(8, 6))
importances.head(15).iloc[::-1].plot(kind='barh', color='#2563EB')
plt.title('Top Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance_10m.png', dpi=150)
plt.show()

#%%
# Diagnostic plots for the champion model
y_true = y_test_raw
y_pred = champion_preds
residuals = y_true - y_pred
metrics = evaluate_predictions(y_true, y_pred)

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

color = '#2563EB'
color_line = '#16A34A'

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_true, y_pred, alpha=0.55, s=35, color=color, edgecolors='white', linewidths=0.4)
lo = min(y_true.min(), y_pred.min()) - 0.5
hi = max(y_true.max(), y_pred.max()) + 0.5
ax1.plot([lo, hi], [lo, hi], '--', color=color_line, lw=1.8)
ax1.set_xlim(lo, hi)
ax1.set_ylim(lo, hi)
ax1.set_xlabel('Actual AU-days')
ax1.set_ylabel('Predicted AU-days')
ax1.set_title('Actual vs Predicted')
ax1.grid(True, alpha=0.25)
ax1.text(
    0.04, 0.96,
    f"R² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.2f}\nMAE = {metrics['MAE']:.2f}",
    transform=ax1.transAxes,
    fontsize=10,
    va='top',
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.8')
)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred, residuals, alpha=0.55, s=35, color=color, edgecolors='white', linewidths=0.4)
ax2.axhline(0, linestyle='--', color=color_line, lw=1.8)
ax2.set_xlabel('Predicted AU-days')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Predicted')
ax2.grid(True, alpha=0.25)

ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(residuals, bins=30, kde=True, color=color, ax=ax3)
ax3.set_title('Residual Distribution')
ax3.set_xlabel('Residual (Actual - Predicted)')

ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(y=residuals, color='#93C5FD', ax=ax4)
ax4.set_title('Residual Boxplot')
ax4.set_ylabel('Residual')

fig.suptitle(f'Diagnostics for {champion_name}', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'diagnostic_plots_10m.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
predictions_df.head()

#%%
print(analysis_df.columns)

#%%
# Saved outputs
print('Saved files:')
for filename in [
    'filtered_df_10m.csv',
    'model_results_10m.csv',
    'test_predictions_10m.csv',
    'feature_importance_10m.png',
    'diagnostic_plots_10m.png',
]:
    print('-', OUTPUT_DIR / filename)

#%%
import numpy as np

# Identify rows where actual_au_days > 2.0 in the predictions_df
outlier_indices_in_predictions_df = predictions_df[predictions_df['actual_au_days'] > 2.0].index

# Map these indices back to the original analysis_df using split['test_idx']
# The test_idx array contains the original indices from analysis_df for the test set.
original_analysis_df_indices_of_outliers = split['test_idx'][outlier_indices_in_predictions_df]

# Get the unique cell_ids for these original indices from analysis_df using .iloc
outlier_cell_ids = analysis_df.iloc[original_analysis_df_indices_of_outliers]['cell_id'].unique()

print("Cell IDs with Actual AU-days > 4 in the test set:")
print(outlier_cell_ids)

# Optionally, also print the actual values to confirm
print("\nActual AU-days values for these outliers:")
print(predictions_df[predictions_df['actual_au_days'] > 2.0]['actual_au_days'])

#%%
print("\nBin IDs for these outliers:")
print(analysis_df.iloc[original_analysis_df_indices_of_outliers]['bin_idx'].unique())

