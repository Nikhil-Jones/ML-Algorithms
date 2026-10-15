from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json, base64, mimetypes
import re

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "normalised_notebooks"
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
