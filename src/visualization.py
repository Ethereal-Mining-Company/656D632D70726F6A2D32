import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_sample_digits(X, y, count=10):
    fig, axes = plt.subplots(1, count, figsize=(12, 2))
    for i, ax in enumerate(axes):
        img = X.iloc[i].values.reshape(8, 8)
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"Digit: {y.iloc[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_df):
    ax = results_df.plot(
        kind="bar",
        x="engine_type",
        y=["cross_val_score", "final_test_score"],
        ylim=(0.7, 1.02),
        figsize=(8, 5),
    )
    ax.set_title("Handwritten Digit Classification: Model Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_custom_predictions(X_custom, y_true, predictions, filenames):
    count = len(X_custom)
    correct_count = sum(int(t == p) for t, p in zip(y_true, predictions))
    fig, axes = plt.subplots(1, count, figsize=(3 * count, 3.4), squeeze=False)
    for ax, (_, row), true_val, pred_val, name in zip(axes.ravel(), X_custom.iterrows(), y_true, predictions, filenames):
        is_right = (true_val == pred_val)
        status = "CORRECT" if is_right else "WRONG"
        color = "green" if is_right else "red"
        ax.imshow(row.values.reshape(8, 8), cmap="gray_r")
        ax.set_title(f"{name}\nTrue: {true_val} | Pred: {pred_val}\n[{status}]", color=color, fontweight="bold")
        ax.axis("off")
    plt.suptitle(f"Custom Handwriting Accuracy: {correct_count}/{count} ({correct_count / count:.0%})", y=1.05, fontweight="bold")
    plt.tight_layout()
    plt.show()