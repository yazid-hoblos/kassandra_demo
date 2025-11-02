import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import contextlib
import time
import os
import glob
import joblib

from core.cell_types import CellTypes
from core.mixer import Mixer
from core.model import DeconvolutionModel
from core.utils import renorm_expressions, tr_to_genes
from core.plotting import print_all_cells_in_one, print_cell_matras, cells_p

st.set_page_config(page_title="Kassandra demo", layout="wide")

st.title("Kassandra — Presentation demo (guided)")
with st.expander("About this demo", expanded=True):
    st.markdown(
        "- Mirrors the same steps as `Model Training.ipynb` with lighter, deterministic subsets for speed.\n"
        "- You can either run a quick pipeline on small real-data slices or show precomputed, full-results artifacts.\n"
        "- Plots are generated like in the notebook (scatter of all cells and per-cell matrices) when running predictions."
    )

# Paths to the real repo data used by the notebook (may be large)
CELLS_EXPR = "data/cells_expr.tsv.tar.gz"
CELLS_ANNOT = "data/cells_samples_annot.tsv.tar.gz"
TUMOR_EXPR = "data/cancer_expr.tsv.tar.gz"
TUMOR_ANNOT = "data/cancer_samples_annot.tsv.tar.gz"
CELL_TYPES_CFG = "configs/cell_types.yaml"
VALIDATION_BLOOD = "data/validation_datasets/blood_expr.tsv"
VALIDATION_CYTOF = "data/validation_datasets/cytometry_df.tsv"

# Deterministic demo subsets (derived from actual data by scripts/make_demo_artifacts.py)
DEMO_DIR = "data/demo_artifacts"
DEMO_CELLS_EXPR = os.path.join(DEMO_DIR, "cells_expr.small.tsv")
DEMO_CELLS_ANNOT = os.path.join(DEMO_DIR, "cells_samples_annot.small.tsv")
DEMO_TUMOR_EXPR = os.path.join(DEMO_DIR, "cancer_expr.small.tsv")
DEMO_TUMOR_ANNOT = os.path.join(DEMO_DIR, "cancer_samples_annot.small.tsv")
DEMO_VAL_BLOOD = os.path.join(DEMO_DIR, "blood_expr.small.tsv")
DEMO_VIC_EXPR = os.path.join(DEMO_DIR, "VIC_expr.small.tsv")

# Precomputed full-run artifacts to mirror notebook outputs (optional)
PRECOMP_DIR = "data/precomputed"
PRECOMP_MODEL = os.path.join(PRECOMP_DIR, "model.joblib")
PRECOMP_METRICS = os.path.join(PRECOMP_DIR, "metrics.csv")
PRECOMP_PREDS = os.path.join(PRECOMP_DIR, "predictions_blood.tsv")
PRECOMP_PLOTS_DIR = os.path.join(PRECOMP_DIR, "plots")


def quiet():
    """Context manager to suppress stdout/stderr for noisy libraries during demo."""
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    return contextlib.redirect_stdout(new_stdout), contextlib.redirect_stderr(new_stderr)


tab = st.sidebar.radio("Step", ["Data curation", "Training data", "Training models", "Validation"])

# Keep state across tabs/reruns
if "model" not in st.session_state:
    st.session_state["model"] = None
if "cell_types" not in st.session_state:
    st.session_state["cell_types"] = None

if tab == "Data curation":
    st.header("1. Data curation")
    st.write("Load expression matrices and apply the cleaning/normalization steps.")

    source = st.selectbox("Choose dataset to preview/load", ["cells_expr", "cancer_expr", "validation_blood"])
    preview_rows = st.slider("Preview rows (genes)", 10, 100, 10)

    try:
        if source == "cells_expr":
            # st.write(f"Previewing `{CELLS_EXPR}` (may be large).")
            df_preview = pd.read_csv(CELLS_EXPR, sep='\t', index_col=0, nrows=preview_rows)
        elif source == "cancer_expr":
            # st.write(f"Previewing `{TUMOR_EXPR}` (may be large).")
            df_preview = pd.read_csv(TUMOR_EXPR, sep='\t', index_col=0, nrows=preview_rows)
        else:
            # st.write(f"Previewing `{VALIDATION_BLOOD}` (may be large).")
            df_preview = pd.read_csv(VALIDATION_BLOOD, sep='\t', index_col=0, nrows=preview_rows)

        st.subheader("Preview (first N genes)")
        st.dataframe(df_preview)
        if st.button("Run normalization"):
            # use renorm_expressions from core.utils (needs full gene list file)
            try:
                renormed = renorm_expressions(df_preview, 'configs/genes_in_expression.txt')
                st.write("Renormalized preview (first rows)")
                st.dataframe(renormed.head())
            except Exception as e:
                st.error(f"Renormalization failed on preview: {e}")

    except FileNotFoundError:
        st.error("Data file not found. Please run scripts/make_demo_artifacts.py to create deterministic small subsets.")
        if os.path.exists(DEMO_VIC_EXPR):
            df = pd.read_csv(DEMO_VIC_EXPR, sep='\t', index_col=0, nrows=preview_rows)
            st.dataframe(df.head())

    # st.markdown("**Note:** For the full cleaning step the notebook uses `tr_to_genes` and `renorm_expressions` (see `core/utils.py`).")

elif tab == "Training data":
    st.header("2. Training data (artificial transcriptomes generation)")
    st.write("Generate pseudobulk (artificial transcriptomes) using `Mixer`. Use small parameters in live demo.")

    quick = st.checkbox("Quick demo mode", value=True)
    num_points = st.number_input("num_points", min_value=10, max_value=10000, value=100)
    num_av = st.number_input("num_av (averages)", min_value=1, max_value=10, value=1)

    try:
        cell_types = CellTypes.load(CELL_TYPES_CFG)
        st.write("Loaded cell types config. Models available:", cell_types.models)
        selected_model = st.selectbox("Choose cell model to generate mixes for", cell_types.models)

        if st.button("Generate pseudobulk"):
            with st.spinner("Generating pseudobulks (may take a few seconds)..."):
                # load small subsets of expression matrices to keep this fast
                try:
                    if quick and os.path.exists(DEMO_CELLS_EXPR) and os.path.exists(DEMO_TUMOR_EXPR):
                        cells_expr = pd.read_csv(DEMO_CELLS_EXPR, sep='\t', index_col=0)
                        cancer_expr = pd.read_csv(DEMO_TUMOR_EXPR, sep='\t', index_col=0)
                        cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                        cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)
                    else:
                        cells_expr = pd.read_csv(CELLS_EXPR, sep='\t', index_col=0)
                        cancer_expr = pd.read_csv(TUMOR_EXPR, sep='\t', index_col=0)
                        cells_annot = pd.read_csv(CELLS_ANNOT, sep='\t', index_col=0)
                        cancer_annot = pd.read_csv(TUMOR_ANNOT, sep='\t', index_col=0)
                        # sample small number of columns (samples)
                        max_samples = 200 if quick else min(2000, cells_expr.shape[1])
                        cells_expr = cells_expr.iloc[:, :max_samples]
                        cancer_expr = cancer_expr.iloc[:, :max_samples]
                except Exception as e:
                    st.warning(f"Could not load repo files quickly; using demo subsets if available. Details: {e}")
                    if os.path.exists(DEMO_CELLS_EXPR) and os.path.exists(DEMO_TUMOR_EXPR):
                        cells_expr = pd.read_csv(DEMO_CELLS_EXPR, sep='\t', index_col=0)
                        cancer_expr = pd.read_csv(DEMO_TUMOR_EXPR, sep='\t', index_col=0)
                        cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                        cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)
                    else:
                        raise

                mixer = Mixer(cell_types=cell_types,
                              cells_expr=cells_expr, cells_annot=cells_annot,
                              tumor_expr=cancer_expr, tumor_annot=cancer_annot,
                              num_av=num_av, num_points=int(num_points))
                expr, values = mixer.generate(selected_model)
                st.success("Pseudobulk generated")
                st.subheader("Generated expression (preview)")
                st.dataframe(expr.iloc[:10, :10])
                st.subheader("Generated ground-truth fractions (values) preview")
                st.dataframe(values.iloc[:, :5])

    except FileNotFoundError as e:
        st.error(f"Cell types config or data missing: {e}")

elif tab == "Training models":
    st.header("3. Training models")
    st.write("Train lightweight models on generated pseudobulks (quick demo) or use a precomputed model for instant results.")
    quick = st.checkbox("Quick demo training (recommended)", value=True)
    num_points = st.number_input("num_points for mixer (when generating)", min_value=10, max_value=10000, value=200)
    num_av = st.number_input("num_av for mixer", min_value=1, max_value=10, value=1)

    colA, colB = st.columns(2)

    if colA.button("Run quick training pipeline"):
        with st.spinner("Running quick pipeline: generate -> train (lightweight)..."):
            try:
                cell_types = CellTypes.load(CELL_TYPES_CFG)
                # load small data as in previous step
                if quick and os.path.exists(DEMO_CELLS_EXPR):
                    cells_expr = pd.read_csv(DEMO_CELLS_EXPR, sep='\t', index_col=0)
                    cancer_expr = pd.read_csv(DEMO_TUMOR_EXPR, sep='\t', index_col=0)
                    cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                    cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)
                else:
                    cells_expr = pd.read_csv(CELLS_EXPR, sep='\t', index_col=0).iloc[:, :200]
                    cancer_expr = pd.read_csv(TUMOR_EXPR, sep='\t', index_col=0).iloc[:, :200]
                    cells_annot = pd.read_csv(CELLS_ANNOT, sep='\t', index_col=0).iloc[:200, :]
                    cancer_annot = pd.read_csv(TUMOR_ANNOT, sep='\t', index_col=0).iloc[:200, :]
            except Exception:
                st.warning("Could not load repo files quickly; using demo subsets (deterministic).")
                cells_expr = pd.read_csv(DEMO_CELLS_EXPR, sep='\t', index_col=0)
                cancer_expr = pd.read_csv(DEMO_TUMOR_EXPR, sep='\t', index_col=0)
                cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)

            mixer = Mixer(cell_types=cell_types,
                          cells_expr=cells_expr, cells_annot=cells_annot,
                          tumor_expr=cancer_expr, tumor_annot=cancer_annot,
                          num_av=int(num_av), num_points=int(num_points))
            model = DeconvolutionModel(cell_types)
            # suppress stdout/stderr during training
            stdout_cm, stderr_cm = quiet()
            try:
                with stdout_cm, stderr_cm:
                    model.fit(mixer)
                st.success("Quick training pipeline finished (models stored in-memory)")
                st.session_state["model"] = model
                st.session_state["cell_types"] = cell_types
            except Exception as e:
                st.error(f"Quick training failed: {e}")

            st.write("You can now run Validation step to predict on sample data.")

    if colB.button("Load precomputed model from disk"):
        if os.path.exists(PRECOMP_MODEL):
            try:
                st.session_state["model"] = joblib.load(PRECOMP_MODEL)
                st.info("Loaded precomputed model from data/precomputed/model.joblib")
            except Exception as e:
                st.error(f"Failed to load precomputed model: {e}")
        else:
            st.warning("data/precomputed/model.joblib not found. See data/precomputed/README.md to generate.")

elif tab == "Validation":
    st.header("4. Validation")
    st.write("Run prediction on a validation dataset (and plot like in the notebook) or show precomputed metrics/plots/predictions.")
    option = st.radio("Mode", ["Run quick prediction", "Show precomputed metrics and plots"]) 

    if option == "Run quick prediction":
        def first_existing(*cands):
            for p in cands:
                if p and os.path.exists(p):
                    return p
            return None

        datasets = []
        # Publication validation set (blood)
        blood_expr = first_existing(VALIDATION_BLOOD, DEMO_VAL_BLOOD)
        blood_cytof = first_existing(VALIDATION_CYTOF)
        if blood_expr and blood_cytof:
            datasets.append(("Publication validation set (blood)", blood_expr, blood_cytof))

        # VIC
        vic_expr = first_existing("data/VIC_expr.tsv", "data/VIC_expr.tsv.tar.gz")
        vic_cytof = first_existing("data/VIC_cytof.tsv", "data/VIC_cytof.tsv.tar.gz")
        if vic_expr and vic_cytof:
            datasets.append(("VIC", vic_expr, vic_cytof))

        # GSE107572
        gse_expr = first_existing("data/GSE107572_expr.tsv.tar.gz", "data/GSE107572_expr.tsv")
        gse_cytof = first_existing("data/GSE107572_cytof.tsv.tar.gz", "data/GSE107572_cytof.tsv")
        if gse_expr and gse_cytof:
            datasets.append(("GSE107572", gse_expr, gse_cytof))

        if not datasets:
            st.error("No paired validation datasets (expr + cytof) found. Try the precomputed view.")
        else:
            label_to_tuple = {lbl: (e, c) for lbl, e, c in datasets}
            choice = st.selectbox("Choose validation dataset", [lbl for lbl, _, _ in datasets])
            expr_path, cytof_path = label_to_tuple[choice]
            st.caption(f"Expression: {expr_path} | Ground truth: {cytof_path}")

            try:
                val_expr = pd.read_csv(expr_path, sep='\t', index_col=0)
                cytof_df = pd.read_csv(cytof_path, sep='\t', index_col=0)
                st.subheader("Validation expression (preview)")
                st.dataframe(val_expr.iloc[:10, :10])
            except Exception as e:
                st.error(f"Failed to load validation files: {e}")
                val_expr, cytof_df = None, None

            if val_expr is not None and cytof_df is not None:
                if st.button("Run prediction and plot (quick)"):
                    model_obj = st.session_state.get("model")
                    if model_obj is None and os.path.exists(PRECOMP_MODEL):
                        try:
                            model_obj = joblib.load(PRECOMP_MODEL)
                            st.info("Using precomputed model from disk.")
                        except Exception as e:
                            st.error(f"Failed to load precomputed model: {e}")
                            model_obj = None
                    if model_obj is None:
                        st.error("No model available. Train (quick) or load a precomputed model first.")
                    else:
                        try:
                            # limit columns for speed if huge
                            cols = min(val_expr.shape[1], 50)
                            val_small = val_expr.iloc[:, :cols]
                            preds = model_obj.predict(val_small) * 100
                            if set(['B_cells','T_cells','NK_cells']).issubset(set(preds.index)):
                                preds.loc['Lymphocytes'] = preds.loc[['B_cells', 'T_cells', 'NK_cells']].sum()

                            st.subheader("Predictions (preview)")
                            st.dataframe(preds.iloc[:, :10])

                            # Align cytof to prediction columns if needed
                            cytof_aligned = cytof_df.loc[preds.index.intersection(cytof_df.index), preds.columns.intersection(cytof_df.columns)]

                            st.subheader("Notebook-like plots")
                            # All cells in one scatter
                            fig1, ax1 = plt.subplots()
                            print_all_cells_in_one(preds, cytof_aligned, ax=ax1, pallete=cells_p,
                                                   title=choice, min_xlim=0, min_ylim=0)
                            st.pyplot(fig1)
                            # Per-cell matrix of scatters
                            axs = print_cell_matras(preds, cytof_aligned, pallete=cells_p, colors_by='index',
                                                    title='', sub_title_font=18, fontsize_title=22,
                                                    subplot_ncols=4, ticks_size=12, wspace=0.4, hspace=0.5,
                                                    min_xlim=0, min_ylim=0)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.error(f"Prediction or plotting failed: {e}")
    else:
        st.subheader("Precomputed metrics and plots")
        if os.path.exists(PRECOMP_METRICS):
            metrics = pd.read_csv(PRECOMP_METRICS)
            st.dataframe(metrics)
            if st.checkbox("Plot training/validation curves"):
                fig, ax = plt.subplots()
                if {'epoch','loss','val_loss'}.issubset(metrics.columns):
                    ax.plot(metrics['epoch'], metrics['loss'], label='train_loss')
                    ax.plot(metrics['epoch'], metrics['val_loss'], label='val_loss')
                else:
                    for col in metrics.columns:
                        if col != 'epoch':
                            ax.plot(metrics['epoch'], metrics[col], label=col)
                ax.legend()
                st.pyplot(fig)
        else:
            st.info("No metrics found at data/precomputed/metrics.csv. See data/precomputed/README.md to generate.")

        # Show precomputed plots if present
        if os.path.isdir(PRECOMP_PLOTS_DIR):
            pngs = sorted(glob.glob(os.path.join(PRECOMP_PLOTS_DIR, "*.png")))
            if pngs:
                st.subheader("Precomputed plots")
                for p in pngs:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
            else:
                st.write("No PNGs found under data/precomputed/plots/ yet.")
        else:
            st.write("data/precomputed/plots/ directory not present.")

        # Show precomputed predictions if present
        if os.path.exists(PRECOMP_PREDS):
            st.subheader("Precomputed predictions (blood)")
            try:
                preds = pd.read_csv(PRECOMP_PREDS, sep='\t', index_col=0)
                st.dataframe(preds.iloc[:, :10])
            except Exception as e:
                st.warning(f"Could not load precomputed predictions: {e}")

# Footer
st.markdown("---")
st.caption("This guided demo mirrors the notebook. For heavy steps it uses deterministic small subsets or precomputed artifacts to stay responsive.")
