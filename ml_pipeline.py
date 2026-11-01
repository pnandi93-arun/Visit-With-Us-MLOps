
import argparse, json, os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tourism.csv"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"
TARGET = "ProdTaken"
ID_COLS = ["CustomerID"]
EXCLUDED = ["TypeofContact", "DurationOfPitch", "NumberOfFollowups",
            "ProductPitched", "PitchSatisfactionScore"]
EXPECTED = ["CustomerID","ProdTaken","Age","TypeofContact","CityTier","DurationOfPitch",
            "Occupation","Gender","NumberOfPersonVisiting","NumberOfFollowups",
            "ProductPitched","PreferredPropertyStar","MaritalStatus","NumberOfTrips",
            "Passport","PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting",
            "Designation","MonthlyIncome"]

def validate():
    if not DATA.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA}")
    df = pd.read_csv(DATA)
    missing = sorted(set(EXPECTED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if not set(df[TARGET].dropna().unique()).issubset({0,1}):
        raise ValueError("ProdTaken must be binary (0/1)")
    print(json.dumps({"rows":len(df),"columns":len(df.columns),
                      "missing_values":int(df.isna().sum().sum()),
                      "duplicate_rows":int(df.duplicated().sum()),
                      "target_counts":df[TARGET].value_counts().to_dict()}, indent=2))

def prepare():
    df = pd.read_csv(DATA)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates()
    X = df.drop(columns=[TARGET] + ID_COLS + EXCLUDED, errors="ignore")
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    ART.mkdir(exist_ok=True)
    train = X_train.copy(); train[TARGET] = y_train.values
    test = X_test.copy(); test[TARGET] = y_test.values
    train.to_csv(ART / "train.csv", index=False)
    test.to_csv(ART / "test.csv", index=False)
    (ART / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2))
    print(f"Saved train={train.shape}, test={test.shape} to {ART}")

def train():
    train_df = pd.read_csv(ART / "train.csv")
    test_df = pd.read_csv(ART / "test.csv")
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET].astype(int)
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET].astype(int)
    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(exclude=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),
                          ("scaler",StandardScaler())]), num),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                          ("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    specs = {
      "Decision Tree": (DecisionTreeClassifier(random_state=42,class_weight="balanced"),
                        {"model__max_depth":[3,6,None],"model__min_samples_leaf":[2,5]}),
      "Random Forest": (RandomForestClassifier(random_state=42,class_weight="balanced",n_jobs=-1),
                        {"model__n_estimators":[150,250],"model__max_depth":[8,None],
                         "model__min_samples_leaf":[2,5]}),
      "Bagging": (BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                                    random_state=42,n_jobs=-1),
                  {"model__n_estimators":[50,100],"model__max_samples":[0.7,1.0]}),
      "AdaBoost": (AdaBoostClassifier(random_state=42),
                   {"model__n_estimators":[50,100],"model__learning_rate":[0.05,0.5,1.0]}),
      "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                            {"model__n_estimators":[75,125],"model__learning_rate":[0.05,0.1],
                             "model__max_depth":[2,3]})
    }
    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    rows=[]; candidates={}
    for name,(est,grid) in specs.items():
        pipe=Pipeline([("preprocessor",prep),("model",est)])
        search=GridSearchCV(pipe,grid,scoring="average_precision",cv=cv,n_jobs=-1,refit=True)
        search.fit(X_train,y_train)
        candidates[name]=search.best_estimator_
        rows.append({"model":name,"cv_pr_auc":float(search.best_score_),
                     "best_params":{k:str(v) for k,v in search.best_params_.items()}})
    results=[]
    for name,model in candidates.items():
        scores=model.predict_proba(X_test)[:,1]
        pred=(scores>=0.5).astype(int)
        results.append({"model":name,"accuracy":float(accuracy_score(y_test,pred)),
          "precision":float(precision_score(y_test,pred,zero_division=0)),
          "recall":float(recall_score(y_test,pred,zero_division=0)),
          "f1":float(f1_score(y_test,pred,zero_division=0)),
          "roc_auc":float(roc_auc_score(y_test,scores)),
          "pr_auc":float(average_precision_score(y_test,scores))})
    best=max(results,key=lambda r:r["pr_auc"])
    model=candidates[best["model"]]
    MODELS.mkdir(exist_ok=True)
    features=list(X_train.columns)
    bundle={"pipeline":model,"threshold":0.50,"model_name":best["model"],
            "feature_columns":features,"target":TARGET,
            "excluded_columns":EXCLUDED,"random_state":42}
    joblib.dump(bundle,MODELS/"best_model.joblib")
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts"/"training_results.json").write_text(json.dumps(
       {"cross_validation":rows,"test_evaluation":results,"selected_model":best},indent=2))
    print(json.dumps({"selected_model":best,"all_results":results},indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["validate","prepare","train"])
    args=parser.parse_args()
    {"validate":validate,"prepare":prepare,"train":train}[args.command]()
