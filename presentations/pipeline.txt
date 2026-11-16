import os, torch
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "mps" if torch.backends.mps.is_available() else "cpu"


import json, joblib, importlib, uuid, keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Input, Reshape, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import accuracy_score
from .utils import load_config

def build_xor_mlp(hidden_units=8, hidden_activation="relu", learning_rate=0.05):
    model = Sequential([Input(shape=(2,)), Dense(hidden_units, activation=hidden_activation), Dense(1, activation="sigmoid")])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cifar_ann(hidden_units=512, learning_rate=0.001, input_dim=3072, num_classes=10):
    model = Sequential([Input(shape=(input_dim,)), Dense(hidden_units, activation="relu"), Dense(hidden_units // 2, activation="relu"), Dense(num_classes, activation="softmax")])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_cifar_cnn(filters=32, dropout_rate=0.25, learning_rate=0.001, input_dim=3072, image_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Input(shape=(input_dim,)),
        Reshape(tuple(image_shape)),
        Conv2D(filters, (3, 3), padding="same", activation="relu"),
        Conv2D(filters, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        Conv2D(filters * 2, (3, 3), padding="same", activation="relu"),
        Conv2D(filters * 2, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def instantiate_model(module_path, class_name, params=None):
    model_class = getattr(importlib.import_module(module_path), class_name)
    return model_class(**params) if params else model_class()

def load_and_split_data(data_path, target_col, seed=42, test_size=0.2, skip_split=False):
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target_col]), df[target_col]
    return (X, X, y, y) if skip_split else train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def _fit_model(model, X_tr, y_tr, fit_params=None):
    model.fit(X_tr, y_tr, **fit_params) if fit_params else model.fit(X_tr, y_tr)

def _predict_classes(model, X_eval):
    preds = model.predict(X_eval, verbose=0) if isinstance(model, keras.Model) else model.predict(X_eval)
    if isinstance(model, keras.Model) and preds.ndim > 1:
        return (preds.flatten() >= 0.5).astype(int) if preds.shape[1] == 1 else np.argmax(preds, axis=1)
    return preds

def _score_model(model, X_val, y_val):
    acc = accuracy_score(y_val, _predict_classes(model, X_val))
    loss = model.evaluate(X_val, y_val, verbose=0)[0] if isinstance(model, keras.Model) else 1.0 - acc
    return acc, loss

def execute_tuning(module_path, class_name, param_grid, X_train, y_train, cv_folds, seed, skip_kfold=False, fit_params=None):
    param_list = list(ParameterGrid(param_grid)) if param_grid else [{}]
    records = []
    for params in param_list:
        keras.utils.set_random_seed(seed)
        sample_model = instantiate_model(module_path, class_name, params)
        full_params = sample_model.get_params() if hasattr(sample_model, "get_params") else {**params, **(fit_params or {})}
        if skip_kfold:
            _fit_model(sample_model, X_train, y_train, fit_params)
            acc, loss = _score_model(sample_model, X_train, y_train)
            scores, losses = [acc], [loss]
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            scores, losses = [],[]
            for train_idx, val_idx in kf.split(X_train, y_train):
                fold_model = instantiate_model(module_path, class_name, params)
                _fit_model(fold_model, X_train.iloc[train_idx], y_train.iloc[train_idx], fit_params)
                acc, loss = _score_model(fold_model, X_train.iloc[val_idx], y_train.iloc[val_idx])
                scores.append(acc); losses.append(loss)
        records.append({"tuned_params": str(params), "mean_cv_accuracy": np.mean(scores), "mean_cv_loss": np.mean(losses), "fold_scores": str(scores), "all_model_parameters": str(full_params)})
    return pd.DataFrame(records), param_list

def save_artifacts(best_engine, payload, log_df, model_dir, log_dir, log_file, is_test, name, run_uuid, seed, cv_folds, test_score):
    run_mode, prefix = ("testing", "test_") if is_test else ("production", "")
    out_dir = os.path.join(model_dir, run_mode)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(payload, os.path.join(out_dir, f"{prefix}{name}_{run_uuid}.pkl"))
    os.makedirs(log_dir, exist_ok=True)
    csv_df = log_df.copy()
    csv_df["run_uuid"], csv_df["test_mode"], csv_df["global_seed"], csv_df["cv_folds"], csv_df["engine_type"], csv_df["final_test_score"] = run_uuid, "Y" if is_test else "N", seed, cv_folds, name, test_score
    csv_path = os.path.join(log_dir, log_file)
    csv_df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

def _build_keras_metadata(feature_names, description, tuning_df, best_idx, run_uuid, name, run_mode, data_path, target_col, seed, cv_folds, best_params, fit_params, test_score, dataset_meta=None):
    raw = {
        "model_metadata": {"run_uuid": run_uuid, "engine_type": name, "run_mode": run_mode, "description": description, "source_data": data_path, "target_variable": target_col, "global_seed": int(seed), "cv_folds": int(cv_folds), "best_tweaks": best_params, "full_winning_parameters": {**best_params, **(fit_params or {})}, "dataset_metadata": dataset_meta or {}},
        "features": {"input_features": feature_names, "feature_count": len(feature_names), "feature_importance": {"info": "Gradient or permutation attribution required for pixel weights in neural network configurations"}},
        "validation_metrics": {"cross_val_score": float(tuning_df.loc[best_idx, "mean_cv_accuracy"]), "mean_cv_loss": float(tuning_df.loc[best_idx, "mean_cv_loss"]), "final_test_score": float(test_score), "all_tuning_runs": tuning_df.to_dict(orient="records")}
    }
    return json.loads(json.dumps(raw, default=str))

def save_keras_artifacts(best_engine, metadata, log_df, model_dir, log_dir, log_file, is_test, name, run_uuid, seed, cv_folds, test_score):
    run_mode, prefix = ("testing", "test_") if is_test else ("production", "")
    out_dir = os.path.join(model_dir, run_mode)
    os.makedirs(out_dir, exist_ok=True)
    best_engine.save(os.path.join(out_dir, f"{prefix}{name}_{run_uuid}.keras"))
    os.makedirs(log_dir, exist_ok=True)
    csv_df = log_df.copy()
    csv_df["all_model_parameters"] = json.dumps(metadata, default=str)
    csv_df["run_uuid"], csv_df["test_mode"], csv_df["global_seed"], csv_df["cv_folds"], csv_df["engine_type"], csv_df["final_test_score"] = run_uuid, "Y" if is_test else "N", seed, cv_folds, name, test_score
    csv_path = os.path.join(log_dir, log_file)
    csv_df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

def run_generalized_pipeline(target_model_name, param_overrides=None, data_path_override=None, target_col_override=None, log_file_override=None):
    config = load_config()
    cfg_model = config["models"].get(target_model_name.lower())
    if cfg_model is None:
        print(f"Operational Alert: Model '{target_model_name}' is not configured in config.yaml.")
        return False
    is_test = config.get("test_mode", False)
    run_mode = "testing" if is_test else "production"
    seed, test_size, folds = config["evaluation_params"]["global_seed"], config["evaluation_params"].get("test_size", 0.2), config["evaluation_params"]["cv_folds"]
    skip_split, skip_kfold, fit_params = cfg_model.get("skip_split", False), cfg_model.get("skip_kfold", False), cfg_model.get("fit_params", None)
    uuid_str = f"run_{uuid.uuid4().hex[:8]}"
    dataset_key = cfg_model.get("dataset_key")
    active_data_path = data_path_override or (config["paths"][dataset_key] if dataset_key else None)
    dataset_meta = config.get("datasets", {}).get(dataset_key, {}) if dataset_key else {}
    active_target_col = target_col_override or cfg_model.get("target_col") or dataset_meta.get("target_col") or config["features"]["target_col"]
    active_log_file = log_file_override or cfg_model.get("log_file") or config["paths"].get("log_file", "experiment_results.csv")
    print(f"\n--- Running [{run_mode.upper()}] Validation For: {target_model_name.upper()} ({uuid_str}) ---\nSource: {active_data_path}")
    param_grid = cfg_model.get("params", {}).copy()
    if param_overrides is not None:
        param_grid.update(param_overrides)
        print(f"Overrides applied: {param_overrides}")
    X_train, X_test, y_train, y_test = load_and_split_data(active_data_path, active_target_col, seed, test_size=test_size, skip_split=skip_split)
    tuning_df, param_list = execute_tuning(cfg_model["module_path"], cfg_model["class_name"], param_grid, X_train, y_train, folds, seed, skip_kfold=skip_kfold, fit_params=fit_params)
    best_idx = tuning_df.sort_values(by=["mean_cv_accuracy", "mean_cv_loss"], ascending=[False, True]).index[0]
    best_params = param_list[best_idx]
    keras.utils.set_random_seed(seed)
    best_engine = instantiate_model(cfg_model["module_path"], cfg_model["class_name"], best_params)
    _fit_model(best_engine, X_train, y_train, fit_params)
    test_score = accuracy_score(y_test, _predict_classes(best_engine, X_test))
    full_params = best_engine.get_params() if hasattr(best_engine, "get_params") else {**best_params, **(fit_params or {})}
    eff_folds = 1 if skip_kfold else folds
    if cfg_model.get("artifact_format") == "keras":
        meta = _build_keras_metadata(list(X_train.columns), cfg_model.get("description", ""), tuning_df, best_idx, uuid_str, target_model_name, run_mode, active_data_path, active_target_col, seed, eff_folds, best_params, fit_params, test_score, dataset_meta)
        save_keras_artifacts(best_engine, meta, tuning_df, config["paths"]["model_dir"], config["paths"]["log_dir"], active_log_file, is_test, target_model_name, uuid_str, seed, eff_folds, test_score)
    else:
        payload = {"model": best_engine, "metadata": {"run_uuid": uuid_str, "engine_type": target_model_name, "run_mode": run_mode, "source_data": active_data_path, "global_seed": seed, "cv_folds": eff_folds, "best_tweaks": best_params, "cross_val_score": tuning_df.loc[best_idx, "mean_cv_accuracy"], "final_test_score": test_score, "all_tuning_runs": tuning_df, "full_winning_parameters": full_params}}
        save_artifacts(best_engine, payload, tuning_df, config["paths"]["model_dir"], config["paths"]["log_dir"], active_log_file, is_test, target_model_name, uuid_str, seed, eff_folds, test_score)
    print(f"Success! Processed {target_model_name.upper()} | Final Accuracy: {test_score:.4f}")
    return True