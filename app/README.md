# ML Capstone Project — Field Decks

A Streamlit app that hosts the existing React + Vite interface as a custom component. The React UI and styling are kept intact; Streamlit handles the app session and passes its API requests to the Python backend in-process.

## Run locally

Requirements: Python 3.10+ and Node.js 18+ for rebuilding the bundled component assets.

### 1. Put notebooks in the expected folders

```text
normalised_notebooks/
├── classification/   # one .ipynb per algorithm
├── regression/       # one .ipynb per algorithm
└── clustering/       # one .ipynb per algorithm
```

The supplied `1_LinearRegression.ipynb` sample is already placed in `normalised_notebooks/regression/`. Add your other notebooks alongside it.

### 2. Install the app dependencies

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app/app.py
```

The React component bundle is included in `app/frontend/dist`, so Node.js is not needed to run or deploy the app. If you edit the existing frontend source, rebuild the bundle with `npm ci && npm run build` from `app/frontend` before deploying.

## Notebook contract

The API reads saved notebook JSON only; it does **not** execute training code. It returns the fixed zero-based cell indices `2, 4, 6, 8, 10, 12, 14`. The UI displays code and saved outputs from cells 10, 12 and 14 under **Code**, and saved outputs from cells 6 and 12 under **Graphs**. Saved PNG/JPEG/SVG and text outputs are rendered when present.

The **Playground** tab loads the exported estimators from the repository's `models/` directory and predicts from the corresponding bundled training data in `data/`. The form only asks for features used by the selected model: regression derives `Item_Category`, `Outlet_Age`, and the engineered numeric features; classification returns the predicted `position`; clustering returns a cluster assignment. Agglomerative clustering has no native `predict` method, so its playground assignment uses the nearest centroid of its saved training clusters.

Algorithm files are discovered by category and filename. Use the names from the brief where possible, for example `1_LinearRegression.ipynb`, `AdaBoost.ipynb`, `RFC.ipynb`, and `1_KmeansClustering.ipynb`. If a notebook cannot be matched, its card remains visible and the detail page explains that the notebook is missing. A future algorithm can be added to `ALGORITHMS` in `backend/main.py`.

## Notes

- `Info` uses a built-in reference library for objective, formulation, limitations and common use cases. It is general algorithm information, not a claim about the exact hyperparameters in your notebook.
- No notebook code is executed, so missing saved outputs remain missing until you save outputs in Jupyter.
- Keep the sibling `data/` and `models/` folders in the deployment repository; the app reads them at runtime.
- Replacing notebooks does not require rebuilding the React component bundle.
