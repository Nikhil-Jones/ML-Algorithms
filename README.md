# ML Algorithms Capstone

An end-to-end machine-learning capstone project with three tracks: BigMart sales regression, football player-position classification, and customer clustering. The Streamlit app hosts the existing React interface and provides access to the exported models and notebooks.

## Repository structure

```text
.
├── README.md
├── PROJECT_STORY_AND_MODEL_GUIDE.md
├── requirements.txt
├── data/
│   ├── Train.csv
│   ├── download_data.py
│   ├── regression/data.csv
│   ├── classification/classification.csv
│   └── clustering/data.csv
├── models/
│   ├── regression/          # 10 exported regressors
│   ├── classification/      # 10 exported classifiers
│   └── clustering/          # K-Means and Agglomerative models
├── notebooks/
│   ├── regression/          # EDA, preprocessing, and 10 model notebooks
│   ├── classification/      # EDA, preprocessing, and 10 model notebooks
│   └── clustering/          # K-Means and Agglomerative analysis
└── app/
    ├── app.py               # Streamlit host and API bridge
    ├── requirements.txt     # Streamlit Cloud dependencies for this entrypoint
    ├── backend/             # Prediction and notebook API logic
    ├── normalised_notebooks/ # Notebook content displayed in the app
    └── frontend/
        ├── src/             # Existing React + Vite interface
        └── dist/            # Built component served by Streamlit
```

## Run locally

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd app/frontend
npm ci
npm run build
cd ../..
streamlit run app/app.py
```

The app needs the `data/`, `models/`, and `app/frontend/dist/` directories available. If you change the React source, rebuild the frontend before launching Streamlit.

## Deploy to Streamlit Community Cloud

- Connect the GitHub repository and select `app/app.py` as the app file.
- Keep `app/requirements.txt`, the data and model files, and the built frontend at `app/frontend/dist/` in the deployed branch.
- Community Cloud installs the app dependencies from `app/requirements.txt` because it sits beside the entrypoint.

## Clustering results

The table combines the K-Means result shown in the project results screenshot with the final Ward-linkage Agglomerative result recorded in its notebook. Higher Silhouette and Calinski–Harabasz scores are generally preferred; lower Davies–Bouldin is preferred.

| Algorithm | Clusters | Configuration | Silhouette Score | Davies–Bouldin Index | Calinski–Harabasz Index |
|---|---:|---|---:|---:|---:|
| K-Means | 4 | `n_init=100`; income and spending score (2 features) | 0.382261 | 0.809177 | 1754.813504 |
| Agglomerative | 4 | Ward linkage; 16 preprocessed features | 0.0923 | 2.1422 | 232.41 |

**Interpretation:** The K-Means values are from the updated two-feature candidate. The Agglomerative notebook's recorded result uses 16 features, including demographics, one-hot encoded categories, and an engineered income-to-spending ratio. Since the models use different feature spaces, their scores are not a strict like-for-like comparison. The Agglomerative notebook also compares linkage methods; its Single-linkage run has higher Silhouette and lower Davies–Bouldin scores, but very low Calinski–Harabasz and a chaining-shaped result. The table reports the notebook's selected final model, Ward linkage.

## Models at a glance

- **Regression:** 10 models estimate `Item_Outlet_Sales`.
- **Classification:** 10 models predict the football player's `position` from player and match statistics.
- **Clustering:** K-Means and Agglomerative Clustering explore customer segments in the customer dataset.

For the project story, preprocessing rationale, and simple model explanations, see [PROJECT_STORY_AND_MODEL_GUIDE.md](PROJECT_STORY_AND_MODEL_GUIDE.md).
