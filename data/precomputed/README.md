Precomputed artifacts for the demo app and to mirror the Model Training.ipynb outputs.

Place the following files here after running the full training (preferably in the notebook) so the Streamlit app can load and display identical results instantly:

- model.joblib — joblib dump of a fitted core.model.DeconvolutionModel instance
- metrics.csv — tabular metrics with at least columns: epoch, loss, val_loss (or any other tracked metrics over epochs)
- predictions_blood.tsv — optional; predictions table on validation_datasets/blood_expr.tsv (rows: cell types, columns: samples)
- plots/ — folder with exported PNGs used in the notebook (scatter plots, correlation matrices, etc.)

Suggested saving snippet (run at the end of training in the notebook or a script):

```python
import joblib, pandas as pd, os
from pathlib import Path
PRECOMP = Path('data/precomputed')
(PRECOMP / 'plots').mkdir(parents=True, exist_ok=True)

# Assuming `model` is a trained DeconvolutionModel, `history` is a list/dict of training metrics
joblib.dump(model, PRECOMP / 'model.joblib')

# If you collected metrics per epoch in a list of dicts
df_metrics = pd.DataFrame(history)
df_metrics.to_csv(PRECOMP / 'metrics.csv', index=False)

# Optional: save some figures
# fig.savefig(PRECOMP / 'plots' / 'my_plot.png', dpi=150)
```

Note: Large artifacts are not tracked in git by default. If you need them versioned, consider Git LFS or publishing release assets.
