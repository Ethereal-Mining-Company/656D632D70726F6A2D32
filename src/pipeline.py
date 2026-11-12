import os, joblib, importlib, uuid
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import accuracy_score
from .utils import load_config

def instantiate_model(module_path, class_name, params=None):
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**params) if params else model_class()

def load_and_split_data(data_path, target_col, seed=42, test_size=0.2):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def execute_kfold_tuning(module_path, class_name, param_grid, X_train, y_train, cv_folds, seed):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    param_list = list(ParameterGrid(param_grid)) if param_grid else [{}]
    records = []
    for params in param_list:
        cv_scores = []
        full_params = instantiate_model(module_path, class_name, params).get_params()
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            fold_model = instantiate_model(module_path, class_name, params)
            fold_model.fit(X_tr, y_tr)
            cv_scores.append(accuracy_score(y_val, fold_model.predict(X_val)))
        records.append({
            "tuned_params": str(params),  # Fixed single column for all models
            "mean_cv_accuracy": np.mean(cv_scores),
            "fold_scores": str(cv_scores),
            "all_model_parameters": str(full_params)
        })
    return pd.DataFrame(records), param_list

def save_artifacts(best_engine, payload, log_df, model_dir, log_dir, is_test, name, run_uuid, seed, cv_folds, test_score):
    run_mode = "testing" if is_test else "production"
    prefix = "test_" if is_test else ""
    out_dir = os.path.join(model_dir, run_mode)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(payload, os.path.join(out_dir, f"{prefix}{name}_{run_uuid}.pkl"))
    os.makedirs(log_dir, exist_ok=True)
    csv_df = log_df.copy()
    csv_df["run_uuid"], csv_df["test_mode"], csv_df["global_seed"], csv_df["cv_folds"], csv_df["engine_type"], csv_df["final_test_score"] = run_uuid, "Y" if is_test else "N", seed, cv_folds, name, test_score
    csv_path = os.path.join(log_dir, "postal_results.csv")
    csv_df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

def run_generalized_pipeline(target_model_name, param_overrides=None, data_path_override=None):
    config = load_config()
    cfg_model = next((m for m in config["experiment_setup"] if m["name"] == target_model_name.lower()), None)
    if cfg_model is None:
        print(f"Operational Alert: Model '{target_model_name}' is not configured in config.yaml.")
        return False
    is_test = config.get("test_mode", False)
    run_mode = "testing" if is_test else "production"
    seed = config["evaluation_params"]["global_seed"]
    folds = config["evaluation_params"]["cv_folds"]
    uuid_str = f"run_{uuid.uuid4().hex[:8]}"
    active_data_path = data_path_override if data_path_override is not None else config["paths"]["dataset"]
    print(f"\n--- Running [{run_mode.upper()}] Validation For: {target_model_name.upper()} ({uuid_str}) ---\nSource: {active_data_path}")
    param_grid = config["model_params"][cfg_model["param_key"]].copy()
    if param_overrides is not None:
        param_grid.update(param_overrides)
        print(f"Overrides applied: {param_overrides}")
    X_train, X_test, y_train, y_test = load_and_split_data(active_data_path, config["features"]["target_col"], seed)
    tuning_df, param_list = execute_kfold_tuning(cfg_model["module_path"], cfg_model["class_name"], param_grid, X_train, y_train, folds, seed)
    best_idx = tuning_df["mean_cv_accuracy"].idxmax()
    best_params = param_list[best_idx]
    best_engine = instantiate_model(cfg_model["module_path"], cfg_model["class_name"], best_params)
    best_engine.fit(X_train, y_train)
    test_score = accuracy_score(y_test, best_engine.predict(X_test))
    payload = {"model": best_engine, "metadata": {"run_uuid": uuid_str, "engine_type": target_model_name, "run_mode": run_mode, "source_data": active_data_path, "global_seed": seed, "cv_folds": folds, "best_tweaks": best_params, "cross_val_score": tuning_df.loc[best_idx, "mean_cv_accuracy"], "final_test_score": test_score, "all_tuning_runs": tuning_df, "full_winning_parameters": best_engine.get_params()}}
    save_artifacts(best_engine, payload, tuning_df, config["paths"]["model_dir"], config["paths"]["log_dir"], is_test, target_model_name, uuid_str, seed, folds, test_score)
    print(f"Success! Processed {target_model_name.upper()}")
    return True