# ANA675 — Machine Learning Assignments and Projects

## Repository Structure

```text
656D632D70726F6A2D32/
├── artifacts/
│   └── production/          # Serialized trained model bundles (.pkl) tagged by run UUID
├── data/                    # Exported datasets used across assignments
├── figures/                 # Generated plots and evaluation figures
├── images/                  # Saved images used in assignments
├── logs/                    # Results and other logs
├── presentations/           # Submission files
├── src/
│   ├── __init__.py          # Package exports
│   ├── pipeline.py          # Data splitting, K-fold tuning, and artifact serialization
│   ├── utils.py             # Config loader, bundle loader, and dataset exporter
│   └── visualization.py     # Reusable plotting and error analysis functions
├── .gitattributes           # Git LFS tracking rules for large artifacts and datasets
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
├── assignments.ipynb        # Notebook for homework assignments
├── config.yaml              # Centralized paths, seeds, and hyperparameter grids
├── main.ipynb               # Primary notebook for running and documenting experiments
└── requirements.txt         # Python dependencies
```

---

## Setup & Installation

1. **Install and initialize Git LFS (required for `.pkl` model bundles and datasets):**
   ```bash
   git lfs install
   ```

2. **Clone the repository and pull LFS files:**
   ```bash
   git clone https://github.com/<your-username>/656D632D70726F6A2D32.git
   cd 656D632D70726F6A2D32
   git lfs pull
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
