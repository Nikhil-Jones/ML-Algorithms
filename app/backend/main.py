from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from functools import lru_cache
import json, base64, mimetypes
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "normalised_notebooks"
PROJECT_ROOT = ROOT.parent
MODELS = PROJECT_ROOT / "models"
DATA = PROJECT_ROOT / "data"
app = FastAPI(title="ML Capstone Notebook API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALGORITHMS = {
 "regression": [
  ("1_LinearRegression","Linear Regression"),("2_RidgeRegression","Ridge Regression"),("3_LassoRegression","Lasso Regression"),("4_ElasticNetRegression","Elastic Net Regression"),("5_PolynomialRegression","Polynomial Regression"),("6_DecisionTree","Decision Tree Regressor"),("7_RandomForest","Random Forest Regressor"),("8_GradientBoosting","Gradient Boosting Regressor"),("9_SupportVector","Support Vector Regression"),("10_KNN","K-Nearest Neighbors")],
 "classification": [
  ("AdaBoost","AdaBoost"),("Bagging","Bagging Classifier"),("DecisionTree","Decision Tree Classifier"),("Gradient_Boosting","Gradient Boosting"),("Knn","K-Nearest Neighbors"),("LogisticRegression","Logistic Regression"),("MLP","Multi-Layer Perceptron"),("Naive_Bayes","Naive Bayes"),("RFC","Random Forest Classifier"),("SVC","Support Vector Classifier")],
 "clustering": [("1_KmeansClustering","K-Means Clustering"),("AgglomerativeClustering","Agglomerative Clustering")]
}
INFO = {
 "Linear Regression": ("Fit a linear relationship between input features and a continuous target.", "Minimize the sum of squared residuals: Σᵢ(yᵢ − (β₀ + xᵢᵀβ))².", "Assumes a linear relationship; sensitive to outliers and multicollinearity.", "Continuous-value prediction and interpretable baselines."),
 "Ridge Regression": ("Linear regression with L2 regularization to shrink coefficients.", "Minimize RSS + αΣⱼβⱼ².", "Usually retains all features; coefficient magnitude is penalized.", "Correlated features and high-dimensional regression."),
 "Lasso Regression": ("Linear regression with L1 regularization that can drive coefficients to zero.", "Minimize RSS + αΣⱼ|βⱼ|.", "Performs embedded feature selection; selection can be unstable for correlated features.", "Sparse models and feature selection."),
 "Elastic Net Regression": ("Combine L1 and L2 regularization in one linear model.", "Minimize RSS + α[rΣ|βⱼ| + (1−r)Σβⱼ²/2].", "Requires tuning regularization strength and L1/L2 mix.", "Correlated predictors where sparsity is also desired."),
 "Polynomial Regression": ("Model nonlinear trends by expanding features into polynomial terms.", "Minimize Σᵢ(yᵢ − (β₀ + Σⱼβⱼxᵢʲ))².", "Higher degrees can overfit and make extrapolation unreliable.", "Smooth nonlinear relationships."),
 "Decision Tree Regressor": ("Partition feature space into regions and predict a target value in each leaf.", "Choose splits that reduce within-node squared error (or another configured impurity).", "Interpretable but prone to high variance without constraints.", "Nonlinear relationships and mixed feature interactions."),
 "Random Forest Regressor": ("Average predictions from many bootstrapped decision trees.", "Regression prediction: f̂(x) = (1/B)ΣᵦTᵦ(x).", "Less variance than a single tree, but less directly interpretable.", "Robust tabular regression baselines."),
 "Gradient Boosting Regressor": ("Build an additive ensemble that iteratively fits residual structure.", "Fₘ(x) = Fₘ₋₁(x) + ηhₘ(x).", "Sensitive to hyperparameters and can overfit if boosting continues too long.", "Tabular data with nonlinear patterns."),
 "Support Vector Regression": ("Fit a function with an ε-insensitive error tube, optionally using kernels.", "Minimize ½||w||² + CΣᵢ(ξᵢ+ξᵢ*), subject to ε-insensitive constraints.", "Scaling and kernel/penalty choices matter; training can be expensive on large datasets.", "Small-to-medium datasets and nonlinear kernel regression."),
 "K-Nearest Neighbors": ("Predict from the labels or values of nearby training examples.", "Regression commonly averages the k nearest targets; distance and weights define neighborhood influence.", "Prediction can be slow; sensitive to feature scaling and irrelevant dimensions.", "Local patterns and simple non-parametric baselines."),
 "AdaBoost": ("Combine weak learners while increasing emphasis on previously misclassified examples.", "F(x) = sign(Σₜ αₜhₜ(x)) for binary classification.", "Can be sensitive to noisy labels and outliers.", "Binary classification with weak base learners."),
 "Bagging Classifier": ("Train estimators on bootstrap samples and aggregate their predictions.", "Classification uses majority vote or averaged class probabilities.", "Reduces variance but does not necessarily reduce bias.", "High-variance learners such as decision trees."),
 "Decision Tree Classifier": ("Split feature space into regions that separate class labels.", "Common split criteria include Gini impurity: 1 − Σₖpₖ², or entropy: −Σₖpₖlog pₖ.", "Can overfit without depth/leaf constraints.", "Interpretable classification and rule extraction."),
 "Gradient Boosting": ("Add weak learners sequentially to reduce a differentiable loss.", "Fₘ(x) = Fₘ₋₁(x) + ηhₘ(x), where each learner follows the loss gradient.", "Hyperparameter-sensitive and sequential to train.", "Strong tabular classification baselines."),
 "Multi-Layer Perceptron": ("Learn nonlinear decision boundaries through stacked neural layers.", "Layer operation: a⁽ˡ⁾ = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾), trained to minimize a task loss.", "Needs scaling and tuning; less interpretable than simple models.", "Complex nonlinear classification."),
 "Logistic Regression": ("Estimate class probabilities using a logistic link.", "p(y=1|x) = 1/(1 + e⁻⁽β₀+xᵀβ⁾); optimize log-loss with configured regularization.", "Linear decision boundary in the chosen feature space.", "Interpretable classification and probability estimates."),
 "Naive Bayes": ("Classify using Bayes' theorem with conditional feature-independence assumptions.", "P(y|x₁,…,xₙ) ∝ P(y)∏ⱼP(xⱼ|y).", "The conditional-independence assumption may be unrealistic.", "Fast text classification and simple high-dimensional baselines."),
 "Random Forest Classifier": ("Aggregate class predictions from many randomized decision trees.", "Predict the majority class or average class probabilities across trees.", "Less interpretable than a single tree; model size can be large.", "Tabular classification with nonlinear feature interactions."),
 "Support Vector Classifier": ("Find a maximum-margin separating boundary, optionally using kernels.", "Minimize ½||w||² + CΣᵢξᵢ subject to yᵢ(wᵀφ(xᵢ)+b) ≥ 1−ξᵢ.", "Feature scaling and kernel choice are important.", "Small-to-medium classification datasets."),
 "K-Means Clustering": ("Partition observations into k clusters around centroid representatives.", "Minimize Σᵢ minⱼ ||xᵢ − μⱼ||².", "Requires k; sensitive to scale, initialization, and non-spherical clusters.", "Exploratory segmentation and compact cluster structures."),
 "Agglomerative Clustering": ("Build a hierarchy by repeatedly merging the closest clusters.", "Merge decisions depend on the selected linkage and distance; no single universal objective applies.", "Can be expensive for large datasets; linkage and distance affect results.", "Hierarchical exploration when nested group structure matters.")
}

REGRESSION_DROP = ["Item_Identifier", "Outlet_Identifier", "Outlet_Establishment_Year", "source"]
CLASSIFICATION_DROP = ["player_id", "player_name", "nationality", "team", "jersey_number", "club_name", "match_id", "match_date", "stadium", "city", "opponent_team", "tournament_stage", "match_result"]
MODEL_FILES = {
    "regression": {"1_LinearRegression":"1_LinearRegression.pkl", "2_RidgeRegression":"2_RidgeRegression.pkl", "3_LassoRegression":"3_LassoRegression.pkl", "4_ElasticNetRegression":"4_ElasticNetRegression.pkl", "5_PolynomialRegression":"5_PolynomialRegression.pkl", "6_DecisionTree":"6_DecisionTree.pkl", "7_RandomForest":"7_RandomForest.pkl", "8_GradientBoosting":"8_GradientBoosting.pkl", "9_SupportVector":"9_SupportVector.pkl", "10_KNN":"10_KNN.pkl"},
    "classification": {key:f"{key}.pkl" for key in ("AdaBoost", "Bagging", "DecisionTree", "Gradient_Boosting", "Knn", "LogisticRegression", "MLP", "Naive_Bayes", "RFC", "SVC")},
    "clustering": {"1_KmeansClustering":"1_KmeansClustering.pkl", "AgglomerativeClustering":"2_Agglomerative_Hierarchial_Clustering.pkl"},
}

def regression_features(frame):
    frame = frame.copy()
    frame["Item_Fat_Content"] = frame["Item_Fat_Content"].replace({"LF":"Low Fat", "low fat":"Low Fat", "reg":"Regular"})
    frame["Item_Weight"] = frame.groupby("Item_Identifier")["Item_Weight"].transform(lambda col: col.fillna(col.mean()))
    frame["Item_Weight"] = frame["Item_Weight"].fillna(frame["Item_Weight"].mean())
    size_modes = frame.groupby("Outlet_Type")["Outlet_Size"].agg(lambda col: col.mode().iloc[0] if not col.mode().empty else "Medium")
    frame["Outlet_Size"] = frame.apply(lambda row: size_modes.get(row["Outlet_Type"], "Medium") if pd.isna(row["Outlet_Size"]) else row["Outlet_Size"], axis=1)
    visibility_means = frame.groupby("Item_Identifier")["Item_Visibility"].mean()
    frame["Item_Visibility"] = frame.apply(lambda row: visibility_means.get(row["Item_Identifier"], row["Item_Visibility"]) if row["Item_Visibility"] == 0 else row["Item_Visibility"], axis=1)
    frame["Item_Category"] = frame["Item_Identifier"].astype(str).str[:2]
    frame.loc[frame["Item_Category"] == "NC", "Item_Fat_Content"] = "Non-Edible"
    frame["Outlet_Age"] = 2013 - pd.to_numeric(frame["Outlet_Establishment_Year"])
    return frame

@lru_cache(maxsize=1)
def regression_context():
    frame = regression_features(pd.read_csv(DATA / "regression" / "data.csv"))
    train = frame[frame["source"] == "train"].copy().drop(columns=REGRESSION_DROP)
    train["Outlet_Size"] = train["Outlet_Size"].map({"Small":0, "Medium":1, "High":2}).fillna(1).astype(int)
    train["Outlet_Location_Type"] = train["Outlet_Location_Type"].map({"Tier 1":0, "Tier 2":1, "Tier 3":2}).fillna(1).astype(int)
    train = pd.get_dummies(train, columns=["Item_Fat_Content", "Item_Type", "Outlet_Type", "Item_Category"], drop_first=True)
    train["Price_per_Weight"] = train["Item_MRP"] / train["Item_Weight"]
    train["Outlet_Size_Num"] = train["Outlet_Size"].map({0:1, 1:2, 2:3})
    train["Outlet_Age_Size"] = train["Outlet_Age"] * train["Outlet_Size_Num"]
    train = train.drop(columns="Outlet_Size_Num")
    X = train.drop(columns="Item_Outlet_Sales")
    X_train, _, _, _ = train_test_split(X, train["Item_Outlet_Sales"], test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return X.columns.tolist(), scaler, frame.groupby("Item_Identifier")["Item_Visibility"].mean().to_dict()

def transform_regression(values):
    columns, scaler, visibility_means = regression_context()
    if float(values["Item_Weight"]) <= 0:
        raise ValueError("Item_Weight must be greater than zero")
    frame = regression_features(pd.DataFrame([values]))
    if float(values["Item_Visibility"]) == 0:
        frame["Item_Visibility"] = frame["Item_Identifier"].map(visibility_means).fillna(frame["Item_Visibility"])
    frame = frame.drop(columns=REGRESSION_DROP, errors="ignore")
    frame["Outlet_Size"] = frame["Outlet_Size"].map({"Small":0, "Medium":1, "High":2}).fillna(1).astype(int)
    frame["Outlet_Location_Type"] = frame["Outlet_Location_Type"].map({"Tier 1":0, "Tier 2":1, "Tier 3":2}).fillna(1).astype(int)
    frame = pd.get_dummies(frame, columns=["Item_Fat_Content", "Item_Type", "Outlet_Type", "Item_Category"], drop_first=True)
    frame["Price_per_Weight"] = frame["Item_MRP"] / frame["Item_Weight"]
    frame["Outlet_Size_Num"] = frame["Outlet_Size"].map({0:1, 1:2, 2:3})
    frame["Outlet_Age_Size"] = frame["Outlet_Age"] * frame["Outlet_Size_Num"]
    frame = frame.drop(columns="Outlet_Size_Num").reindex(columns=columns, fill_value=0)
    return scaler.transform(frame)

@lru_cache(maxsize=1)
def classification_context():
    frame = pd.read_csv(DATA / "classification" / "classification.csv").sample(n=10000, random_state=42).reset_index(drop=True)
    for col in ["shots", "tackles", "defensive_actions", "accelerations", "decelerations"]:
        frame[col] = frame[col].clip(frame[col].quantile(0.01), frame[col].quantile(0.99))
    labels = LabelEncoder().fit(frame["position"])
    y = labels.transform(frame["position"])
    X = frame.drop(columns=CLASSIFICATION_DROP + ["position"]).copy()
    X["preferred_foot"] = X["preferred_foot"].map({"Left":0, "Right":1})
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X.columns.tolist(), X_train.dtypes.astype(str).to_dict(), StandardScaler().fit(X_train), labels

def transform_classification(values):
    columns, _, scaler, _ = classification_context()
    frame = pd.DataFrame([values]).reindex(columns=columns)
    frame["preferred_foot"] = frame["preferred_foot"].map({"Left":0, "Right":1})
    return scaler.transform(frame)

@lru_cache(maxsize=1)
def clustering_context():
    frame = pd.read_csv(DATA / "clustering" / "data.csv", keep_default_na=False)
    numeric = ["Age", "Annual Income ($)", "Spending Score (1-100)", "Work Experience", "Family Size"]
    categorical = ["Gender", "Profession"]
    preprocessor = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    preprocessor.fit(frame.drop(columns=["CustomerID"]))
    agglom_features = pd.DataFrame({"Age":frame["Age"], "Annual Income ($)":frame["Annual Income ($)"], "Spending Score (1-100)":frame["Spending Score (1-100)"]})
    agglom_features["Income_to_Spending_Ratio"] = agglom_features["Annual Income ($)"] / (agglom_features["Spending Score (1-100)"] + 1)
    agglom_columns = ["Age", "Annual Income ($)", "Spending Score (1-100)", "Income_to_Spending_Ratio"]
    scaler = StandardScaler().fit(agglom_features[agglom_columns])
    return preprocessor, scaler, agglom_features, agglom_columns

def playground_fields(category, algorithm_id):
    if category == "regression":
        frame = pd.read_csv(DATA / "regression" / "data.csv")
        return [
            {"name":"Item_Identifier", "type":"text", "default":"FDA15"}, {"name":"Item_Weight", "type":"number", "default":float(frame["Item_Weight"].median())},
            {"name":"Item_Fat_Content", "type":"select", "options":["Low Fat", "Regular"], "default":"Low Fat"}, {"name":"Item_Visibility", "type":"number", "default":float(frame["Item_Visibility"].median())},
            {"name":"Item_Type", "type":"select", "options":sorted(frame["Item_Type"].dropna().unique().tolist()), "default":str(frame["Item_Type"].mode().iloc[0])}, {"name":"Item_MRP", "type":"number", "default":float(frame["Item_MRP"].median())},
            {"name":"Outlet_Establishment_Year", "type":"number", "default":int(frame["Outlet_Establishment_Year"].mode().iloc[0])},
            {"name":"Outlet_Size", "type":"select", "options":["Small", "Medium", "High"], "default":"Medium"}, {"name":"Outlet_Location_Type", "type":"select", "options":["Tier 1", "Tier 2", "Tier 3"], "default":"Tier 1"},
            {"name":"Outlet_Type", "type":"select", "options":sorted(frame["Outlet_Type"].dropna().unique().tolist()), "default":str(frame["Outlet_Type"].mode().iloc[0])},
        ]
    if category == "classification":
        columns, _, _, _ = classification_context()
        data = pd.read_csv(DATA / "classification" / "classification.csv")
        fields=[]
        for col in columns:
            if col == "preferred_foot": fields.append({"name":col,"type":"select","options":["Left","Right"],"default":"Right"})
            else: fields.append({"name":col,"type":"number","default":float(data[col].median())})
        return fields
    if category == "clustering":
        data=pd.read_csv(DATA / "clustering" / "data.csv", keep_default_na=False)
        fields = [{"name":"Age","type":"number","default":float(data["Age"].median())}, {"name":"Annual Income ($)","type":"number","default":float(data["Annual Income ($)"].median())},
            {"name":"Spending Score (1-100)","type":"number","default":float(data["Spending Score (1-100)"].median())}]
        if algorithm_id == "1_KmeansClustering":
            fields.extend([{"name":"Work Experience","type":"number","default":float(data["Work Experience"].median())}, {"name":"Family Size","type":"number","default":float(data["Family Size"].median())}])
            professions = ["Unknown" if value == "" else value for value in sorted(data["Profession"].unique().tolist())]
            fields.insert(0, {"name":"Profession","type":"select","options":professions,"default":str(data["Profession"].mode().iloc[0] or "Unknown")})
            fields.insert(0, {"name":"Gender","type":"select","options":sorted(data["Gender"].unique().tolist()),"default":str(data["Gender"].mode().iloc[0])})
        return fields
    raise HTTPException(400, "Invalid category")

@lru_cache(maxsize=None)
def load_model(category, algorithm_id):
    filename = MODEL_FILES.get(category, {}).get(algorithm_id)
    if not filename:
        raise HTTPException(404, "No exported model is configured for this algorithm")
    path = MODELS / category / filename
    if not path.is_file():
        raise HTTPException(404, f"Exported model not found: {filename}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise HTTPException(500, f"Could not load model artifact {filename}: {exc}")

@app.get('/api/playground/{category}/{algorithm_id}')
def playground_schema(category: str, algorithm_id: str):
    if algorithm_id not in MODEL_FILES.get(category, {}):
        raise HTTPException(404, "Unknown algorithm")
    notes = {
        "regression":"Item_Outlet_Sales and source are not inputs. Item_Category, Outlet_Age, and other engineered fields are calculated from the form values.",
        "classification":"The model predicts position. Identifier and match context columns excluded during training are not requested.",
        "clustering":"Only the features used by this saved model are requested; unused identifiers and extra dataset columns are omitted.",
    }
    note = notes[category]
    if category == "clustering" and algorithm_id == "AgglomerativeClustering":
        note = "The model uses age, annual income, and spending score; unused identifiers and extra dataset columns are omitted."
    return {"category":category, "algorithm":algorithm_id, "fields":playground_fields(category, algorithm_id), "note":note}

@app.post('/api/predict/{category}/{algorithm_id}')
def predict(category: str, algorithm_id: str, values: dict):
    if algorithm_id not in MODEL_FILES.get(category, {}):
        raise HTTPException(404, "Unknown algorithm")
    model = load_model(category, algorithm_id)
    try:
        if category == "regression":
            expected = {field["name"] for field in playground_fields(category, algorithm_id)}
            missing = expected - values.keys()
            if missing: raise HTTPException(422, f"Missing inputs: {', '.join(sorted(missing))}")
            transformed = transform_regression(values)
            if algorithm_id == "5_PolynomialRegression":
                transformed = PolynomialFeatures(degree=2, include_bias=False).fit_transform(transformed)
            result = float(model.predict(transformed)[0])
            return {"prediction":result, "label":f"{result:,.2f}", "target":"Item_Outlet_Sales"}
        if category == "classification":
            columns, _, _, labels = classification_context()
            missing = set(columns) - values.keys()
            if missing: raise HTTPException(422, f"Missing inputs: {', '.join(sorted(missing))}")
            result = int(model.predict(transform_classification(values))[0])
            label = str(labels.inverse_transform([result])[0])
            response = {"prediction":label, "label":label, "target":"position"}
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transform_classification(values))[0]
                response["confidence"] = float(np.max(probabilities))
            return response
        if algorithm_id == "1_KmeansClustering":
            preprocessor, _, _, _ = clustering_context()
            sample = pd.DataFrame([values])[["Gender", "Age", "Annual Income ($)", "Spending Score (1-100)", "Profession", "Work Experience", "Family Size"]]
            sample["Profession"] = sample["Profession"].replace({"Unknown":""})
            label = int(model.predict(preprocessor.transform(sample))[0])
        else:
            _, scaler, training_features, columns = clustering_context()
            labels = np.asarray(model.labels_)
            if len(labels) != len(training_features):
                raise HTTPException(500, "Agglomerative model labels do not match the bundled training data")
            centers = np.vstack([scaler.transform(training_features[labels == i][columns]).mean(axis=0) for i in range(model.n_clusters)])
            row = pd.DataFrame([{key:values[key] for key in ("Age", "Annual Income ($)", "Spending Score (1-100)")}])
            row["Income_to_Spending_Ratio"] = row["Annual Income ($)"] / (row["Spending Score (1-100)"] + 1)
            transformed = scaler.transform(row[columns])[0]
            label = int(np.argmin(np.linalg.norm(centers - transformed, axis=1)))
        response = {"prediction":label, "label":f"Cluster {label}", "target":"cluster"}
        if algorithm_id == "AgglomerativeClustering":
            response["note"] = "Agglomerative clustering has no native predict method; this assigns the nearest training-cluster centroid."
        return response
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise HTTPException(422, f"Invalid playground input: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")

def output_item(out):
    data = out.get("data", {})
    rendered=[]
    for mime, val in data.items():
        vals = val if isinstance(val, list) else [val]
        if mime == "image/png": rendered.append({"mime":mime,"data":"data:image/png;base64,"+"".join(vals)})
        elif mime == "image/jpeg": rendered.append({"mime":mime,"data":"data:image/jpeg;base64,"+"".join(vals)})
        elif mime == "image/svg+xml": rendered.append({"mime":mime,"data":"".join(vals)})
        elif mime in ("text/plain", "text/html", "application/json"):
            rendered.append({"mime":mime,"data":"".join(str(x) for x in vals)})
    if out.get("text"):
        rendered.append({"mime":"text/plain","data":"".join(out["text"])})
    return {"output_type":out.get("output_type"),"data":rendered,"ename":out.get("ename"),"evalue":out.get("evalue")}

def read_cell(cells, idx):
    if idx >= len(cells): return {"index":idx,"cell_type":"missing","source":"","outputs":[]}
    c=cells[idx]
    return {"index":idx,"cell_type":c.get("cell_type"),"source":"".join(c.get("source",[])),"outputs":[output_item(o) for o in c.get("outputs",[])]}

@app.get('/api/health')
def health(): return {"ok":True,"notebooksRoot":str(NOTEBOOKS)}

@app.get('/api/algorithms')
def algorithms():
    result={}
    for category, items in ALGORITHMS.items():
        result[category]=[]
        folder=NOTEBOOKS/category
        files=list(folder.glob('*.ipynb')) if folder.exists() else []
        for stem, label in items:
            candidates=[p for p in files if p.stem.lower()==stem.lower() or p.stem.lower().startswith(stem.lower()+"_") or stem.lower().startswith(p.stem.lower()+"_")]
            # regression filenames use numbered prefixes; match on normalized suffix/name tokens
            if not candidates:
                key=re.sub(r'[^a-z0-9]','',label.lower().replace('regressor','').replace('classifier',''))
                candidates=[p for p in files if key and key in re.sub(r'[^a-z0-9]','',p.stem.lower())]
            result[category].append({"id":stem,"name":label,"notebook":candidates[0].name if candidates else None,"available":bool(candidates)})
    return result

@app.get('/api/notebook/{category}/{filename}')
def notebook(category:str, filename:str):
    if category not in ALGORITHMS or '/' in filename or not filename.endswith('.ipynb'): raise HTTPException(400,'Invalid notebook path')
    path=(NOTEBOOKS/category/filename).resolve()
    if not path.is_relative_to(NOTEBOOKS.resolve()) or not path.exists(): raise HTTPException(404,'Notebook not found')
    try: nb=json.loads(path.read_text(encoding='utf-8'))
    except Exception as e: raise HTTPException(500,'Could not parse notebook: '+str(e))
    cells=nb.get('cells',[])
    return {"filename":filename,"category":category,"cells":{str(i):read_cell(cells,i) for i in (2,4,6,8,10,12,14)},"cellCount":len(cells)}

@app.get('/api/info/{name}')
def info(name:str):
    from urllib.parse import unquote
    name=unquote(name)
    # Match display names; fallback to a useful generic description.
    v=INFO.get(name)
    if not v: return {"objective":"Information has not been added for this algorithm yet.","function":"See the notebook for the implementation and configured loss/criterion.","limitations":"Depends on data, preprocessing and hyperparameters.","uses":"Review the notebook's training and evaluation results."}
    return {"objective":v[0],"function":v[1],"limitations":v[2],"uses":v[3]}
