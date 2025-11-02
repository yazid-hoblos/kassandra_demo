Kassandra demo (Render-ready)

This folder is a minimal app scaffold for deploying the Kassandra Streamlit demo on Render.

Required contents for a complete deploy:

- streamlit_app.py — the Streamlit app (imports from `core.*`)
- requirements.txt — Python deps (streamlit, lightgbm, etc.)
- runtime.txt — Python version (3.10.13)
- render.yaml — Render service config (build + start commands)
- core/ — copy the Kassandra core package here (cell_types.py, mixer.py, model.py, utils.py, plotting.py, __init__.py)
- configs/
  - cell_types.yaml
  - boosting_params/
    - lgb_parameters_first_step.tsv
    - lgb_parameters_second_step.tsv
- data/
  - demo_artifacts/ — deterministic small subsets created from real data (optional but recommended)
  - precomputed/ — artifacts exported from the notebook (model.joblib, metrics.csv, plots/*.png, predictions_blood.tsv)
  - validation_datasets/
    - blood_expr.tsv (optional, for plots in Validation)
    - cytometry_df.tsv (optional, for plots in Validation)

Quick checklist before pushing:

- [ ] `core/` exists inside this folder (app imports from `core.*`)
- [ ] `configs/` exists with `cell_types.yaml` and boosting params
- [ ] `data/precomputed/` has at least `model.joblib` and `metrics.csv`
- [ ] (Optional) `data/demo_artifacts/` and `data/validation_datasets/` with the two small TSVs
- [ ] `render.yaml` points to `streamlit_app.py` (already set)

Alternative: Deploy from repo root

The repository root already contains an `app/streamlit_app.py` and a Render config. You can deploy from the root instead of this folder to avoid duplicating `core/`, `configs/`, and `data/`.
