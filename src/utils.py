import os, joblib, yaml, importlib
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at: '{config_path}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_bundle(model_folder, run_uuid, name, is_test=False):
    run_mode = "testing" if is_test else "production"
    prefix = "test_" if is_test else ""
    path = os.path.join(model_folder, run_mode, f"{prefix}{name}_{run_uuid}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved bundle found at {path}")
    return joblib.load(path)

def export_sklearn_dataset(loader_name, target_file_name, target_col_name="target"):
    os.makedirs("data", exist_ok=True)
    module = importlib.import_module("sklearn.datasets")
    loader_func = getattr(module, loader_name)
    dataset_obj = loader_func()
    df = pd.DataFrame(dataset_obj.data)
    df[target_col_name] = dataset_obj.target
    csv_out_path = os.path.join("data", target_file_name)
    df.to_csv(csv_out_path, index=False)
    print(f"Success! Exported '{loader_name}' cleanly into '{csv_out_path}'")
    return csv_out_path

def load_and_preprocess_images(folder_path="images"):
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    filenames = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(exts))
    rows = list()
    y_true = list()
    for name in filenames:
        stem, _ = os.path.splitext(name)
        word = stem.lower().split("_")[0].split("-")[0]
        y_true.append(word_to_num[word] if word in word_to_num else int(word))
        img = ImageOps.invert(ImageOps.exif_transpose(Image.open(os.path.join(folder_path, name))).convert("L"))
        arr = np.array(img, dtype=float)
        arr = np.where(arr > arr.mean() + 45, 255.0, 0.0)
        ys, xs = np.where(arr > 0)
        if xs.size:
            crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape
            size = int(max(h, w) * 1.25)
            sq = np.zeros((size, size))
            y0, x0 = (size - h) // 2, (size - w) // 2
            sq[y0:y0 + h, x0:x0 + w] = crop
            img = Image.fromarray(sq.astype(np.uint8)).resize((64, 64), Image.Resampling.BILINEAR).filter(ImageFilter.MaxFilter(3))
        a8 = np.array(img.resize((8, 8), Image.Resampling.BILINEAR), dtype=float)
        rows.append(((a8 / a8.max()) * 16.0 if a8.max() > 0 else a8).reshape(64))
    return pd.DataFrame(rows, columns=[str(i) for i in range(64)]), pd.Series(y_true, name="true_digit"), filenames