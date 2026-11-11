import os, joblib, yaml

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_bundle(model_folder):
    path = os.path.join(model_folder, "engine.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved bundle found at {path}")
    return joblib.load(path)