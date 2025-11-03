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
import sys
import pickle
import warnings
import gc

from core.cell_types import CellTypes
from core.mixer import Mixer
from core.model import DeconvolutionModel
from core.utils import renorm_expressions, tr_to_genes
from core.plotting import print_all_cells_in_one, print_cell_matras, cells_p

st.set_page_config(page_title="Kassandra demo", layout="wide")

st.title("Kassandra — Presentation demo")
# with st.expander("About this demo", expanded=True):
#     st.markdown(
#         "- Mirrors the same steps as `Model Training.ipynb` with lighter, deterministic subsets for speed.\n"
#         "- You can either run a quick pipeline on small real-data slices or show precomputed, full-results artifacts.\n"
#         "- Plots are generated like in the notebook (scatter of all cells and per-cell matrices) when running predictions."
#     )

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

# Lite-mode boosting params (tiny models for 512MB instances)
LITE_BP_DIR = "configs/boosting_params_lite"
LITE_BP1 = os.path.join(LITE_BP_DIR, "lgb_parameters_first_step.tsv")
LITE_BP2 = os.path.join(LITE_BP_DIR, "lgb_parameters_second_step.tsv")


def is_lite_env() -> bool:
    """Detect low-memory environment (e.g., Render free) and force lite mode."""
    # Render sets a few env vars; also allow explicit opt-in via KASSANDRA_LITE
    return bool(os.environ.get("KASSANDRA_LITE") or os.environ.get("RENDER") or os.environ.get("RENDER_EXTERNAL_URL"))


@st.cache_data(show_spinner=False)
def load_tsv(path: str, index_col=0, nrows=None, dtype="float32"):
    df = pd.read_csv(path, sep='\t', index_col=index_col, nrows=nrows)
    # cast numeric frames to float32 to save memory, ignore fails on mixed dtypes
    try:
        return df.astype(dtype)
    except Exception:
        return df


def quiet():
    """Context manager to suppress stdout/stderr for noisy libraries during demo."""
    new_stdout, new_stderr = io.StringIO(), io.StringIO()
    return contextlib.redirect_stdout(new_stdout), contextlib.redirect_stderr(new_stderr)


tab = st.sidebar.radio("Step", ["Data curation", "Artificial Transcriptomes", "Model Training", "Validation"])

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

elif tab == "Artificial Transcriptomes":
    st.header("2. Training data (artificial transcriptomes generation)")
    st.write("Generate pseudobulk (artificial transcriptomes) using `Mixer`.")

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

elif tab == "Model Training":
    st.header("3. Training models")
    st.write("Train lightweight models on generated pseudobulks (quick demo).")
    quick_default = True
    lite = is_lite_env()
    if lite:
        st.info("Low-memory environment detected. Enabling lite training mode (tiny models, fewer points).")
    quick = st.checkbox("Quick demo training (recommended)", value=quick_default)
    # Cap aggressively in lite mode
    max_pts = 2000 if not lite else 200
    default_pts = 200 if not lite else 50
    num_points = st.number_input("num_points for mixer (when generating)", min_value=10, max_value=max_pts, value=default_pts)
    num_av = st.number_input("num_av for mixer", min_value=1, max_value=10, value=1)

    colA, colB = st.columns(2)

    if colA.button("Run quick training pipeline"):
        with st.spinner("Running quick pipeline: generate -> train (lightweight)..."):
            try:
                cell_types = CellTypes.load(CELL_TYPES_CFG)
                # load small data as in previous step
                if quick and os.path.exists(DEMO_CELLS_EXPR):
                    cells_expr = load_tsv(DEMO_CELLS_EXPR)
                    cancer_expr = load_tsv(DEMO_TUMOR_EXPR)
                    cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                    cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)
                else:
                    cells_expr = load_tsv(CELLS_EXPR).iloc[:, :200]
                    cancer_expr = load_tsv(TUMOR_EXPR).iloc[:, :200]
                    cells_annot = pd.read_csv(CELLS_ANNOT, sep='\t', index_col=0).iloc[:200, :]
                    cancer_annot = pd.read_csv(TUMOR_ANNOT, sep='\t', index_col=0).iloc[:200, :]
            except Exception:
                st.warning("Could not load repo files quickly; using demo subsets (deterministic).")
                cells_expr = load_tsv(DEMO_CELLS_EXPR)
                cancer_expr = load_tsv(DEMO_TUMOR_EXPR)
                cells_annot = pd.read_csv(DEMO_CELLS_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_CELLS_ANNOT) else pd.DataFrame(index=cells_expr.columns)
                cancer_annot = pd.read_csv(DEMO_TUMOR_ANNOT, sep='\t', index_col=0) if os.path.exists(DEMO_TUMOR_ANNOT) else pd.DataFrame(index=cancer_expr.columns)

            # further cap points in lite mode to avoid OOM
            eff_points = int(min(num_points, 50 if lite else num_points))
            mixer = Mixer(cell_types=cell_types,
                          cells_expr=cells_expr, cells_annot=cells_annot,
                          tumor_expr=cancer_expr, tumor_annot=cancer_annot,
                          num_av=int(num_av), num_points=eff_points)
            # Use tiny LightGBM params in lite environments
            if lite and os.path.exists(LITE_BP1) and os.path.exists(LITE_BP2):
                model = DeconvolutionModel(cell_types,
                                           boosting_params_first_step=LITE_BP1,
                                           boosting_params_second_step=LITE_BP2)
            else:
                model = DeconvolutionModel(cell_types)
            # suppress stdout/stderr during training
            stdout_cm, stderr_cm = quiet()
            try:
                with stdout_cm, stderr_cm:
                    model.fit(mixer)
                st.success("Quick training pipeline finished (models stored in-memory)")
                st.session_state["model"] = model
                st.session_state["cell_types"] = cell_types
                # free up memory used by training dataframes
                del cells_expr, cancer_expr, cells_annot, cancer_annot, mixer
                gc.collect()
            except Exception as e:
                st.error(f"Quick training failed: {e}")

            st.write("You can now run Validation step to predict on sample data.")

    # Note: Loading a precomputed model from disk is intentionally disabled in the hosted demo
    # because large model files can fail to load reliably in constrained/cloud environments.
    # To use a precomputed model, copy `data/precomputed/model.joblib` into the repo locally
    # and run the notebook workflow instead.

elif tab == "Validation":
    st.header("4. Validation")
    st.write("Run prediction on a validation dataset or show precomputed metrics.")
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
                    if model_obj is None:
                        st.error("No model available in memory. Please run 'Run quick training pipeline' in the Training models tab to create a model for prediction.")
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

                            st.subheader("Validation Plots")
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
            st.info("No metrics found at data/precomputed/metrics.csv.")

        # Show precomputed plots if present
        if os.path.exists(PRECOMP_PREDS) and os.path.exists(VALIDATION_CYTOF):
            st.subheader("Precomputed visualization")
            # Load and display plots
            try:
                preds = pd.read_csv(PRECOMP_PREDS, sep='\t', index_col=0)
                cytof_df = pd.read_csv(VALIDATION_CYTOF, sep='\t', index_col=0)
                
                # Align cytof to prediction columns
                cytof_aligned = cytof_df.loc[preds.index.intersection(cytof_df.index), 
                                           preds.columns.intersection(cytof_df.columns)]
                
                # Generate notebook-style plots
                # All cells in one scatter
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                print_all_cells_in_one(preds, cytof_aligned, ax=ax1, pallete=cells_p,
                                     title="Precomputed Results", min_xlim=0, min_ylim=0)
                st.pyplot(fig1)
                
                # Per-cell matrix of scatters
                st.subheader("Per-cell correlation matrix")
                axs = print_cell_matras(preds, cytof_aligned, pallete=cells_p, colors_by='index',
                                      title='', sub_title_font=18, fontsize_title=22,
                                      subplot_ncols=4, ticks_size=12, wspace=0.4, hspace=0.5,
                                      min_xlim=0, min_ylim=0)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error(f"Could not generate precomputed plots: {e}")
            
            # Try to save plots for future use, but don't fail UI if saving fails
            if 'fig1' in locals():
                try:
                    os.makedirs(PRECOMP_PLOTS_DIR, exist_ok=True)
                    with contextlib.suppress(Exception):
                        plt.figure(fig1.number)
                        plt.savefig(os.path.join(PRECOMP_PLOTS_DIR, 'all_cells_scatter.png'), bbox_inches='tight')
                    with contextlib.suppress(Exception):
                        target_fig = axs[0,0].figure if hasattr(axs, '__getitem__') else plt.gcf()
                        plt.figure(target_fig.number)
                        plt.savefig(os.path.join(PRECOMP_PLOTS_DIR, 'cell_correlation_matrix.png'), bbox_inches='tight')
                except Exception as e:
                    st.info(f"Displayed plots but couldn't save to {PRECOMP_PLOTS_DIR}: {e}")
                
        elif os.path.isdir(PRECOMP_PLOTS_DIR):
            # Fallback to just showing saved plots
            pngs = sorted(glob.glob(os.path.join(PRECOMP_PLOTS_DIR, "*.png")))
            if pngs:
                st.subheader("Precomputed plots")
                for p in pngs:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
            else:
                st.write("No precomputed plots available yet. Run prediction with the model first.")

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
st.caption("For heavy steps deterministic small subsets or precomputed artifacts are used to maintain responsiveness.")
