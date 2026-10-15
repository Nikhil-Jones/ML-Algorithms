# ML Capstone Project — Field Decks

A React + Vite frontend and FastAPI notebook reader for your standardized ML notebooks. The visual treatment is an original, responsive editorial/trading-card design inspired by the provided references; the reference skills are included in `reference/`.

## Run locally

Requirements: Python 3.10+ and Node.js 18+.

### 1. Put notebooks in the expected folders

```text
normalised_notebooks/
├── classification/   # one .ipynb per algorithm
├── regression/       # one .ipynb per algorithm
└── clustering/       # one .ipynb per algorithm
```

The supplied `1_LinearRegression.ipynb` sample is already placed in `normalised_notebooks/regression/`. Add your other notebooks alongside it.

### 2. Start the API

From the extracted project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### 3. Start the frontend (second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (usually http://localhost:5173).

## Notebook contract

The API reads saved notebook JSON only; it does **not** execute training code. It returns the fixed zero-based cell indices `2, 4, 6, 8, 10, 12, 14`. The UI displays code and saved outputs from cells 10, 12 and 14 under **Code**, and saved outputs from cells 6 and 12 under **Graphs**. Saved PNG/JPEG/SVG and text outputs are rendered when present.

Algorithm files are discovered by category and filename. Use the names from the brief where possible, for example `1_LinearRegression.ipynb`, `AdaBoost.ipynb`, `RFC.ipynb`, and `1_KmeansClustering.ipynb`. If a notebook cannot be matched, its card remains visible and the detail page explains that the notebook is missing. A future algorithm can be added to `ALGORITHMS` in `backend/main.py`.

## Notes

- `Info` uses a built-in reference library for objective, formulation, limitations and common use cases. It is general algorithm information, not a claim about the exact hyperparameters in your notebook.
- No notebook code is executed, so missing saved outputs remain missing until you save outputs in Jupyter.
- `normalised_notebooks/` is intentionally kept separate from the frontend build, so replacing notebooks does not require rebuilding React.
