# Visit with Us – Wellness Tourism MLOps

## Overview
Pre-contact customer purchase-propensity model using the supplied `ProdTaken` label as a proxy for historical package purchase behaviour. It is not a Wellness-package-specific label.

## Run locally
```bash
pip install -r requirements.txt
python src/ml_pipeline.py validate
python src/ml_pipeline.py prepare
python src/ml_pipeline.py train
streamlit run app/streamlit_app.py
```

## GitHub Actions
Push the repository to GitHub with the default branch named `main`. Enable Actions and ensure the repository Actions setting permits read/write access to repository contents. The workflow validates the CSV, creates train/test files, passes them as an artifact, trains and evaluates models, and commits the updated model and evaluation JSON.

## Streamlit Community Cloud
Create a public GitHub repository, deploy `app/streamlit_app.py` from the repository root, and ensure `models/best_model.joblib` is committed. Add the public Streamlit URL here after deployment.

## Data governance and prediction timing
The primary model excludes contact-channel and pitch interaction fields because the intended prediction is made before contacting the customer. `ProdTaken` is a historical purchase label and is only a proxy for Wellness package propensity.
