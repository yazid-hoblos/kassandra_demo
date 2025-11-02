import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    ROOT / 'streamlit_app.py',
    ROOT / 'requirements.txt',
    ROOT / 'render.yaml',
    ROOT / 'runtime.txt',
    ROOT / 'core',
    ROOT / 'configs' / 'cell_types.yaml',
    ROOT / 'configs' / 'boosting_params' / 'lgb_parameters_first_step.tsv',
    ROOT / 'configs' / 'boosting_params' / 'lgb_parameters_second_step.tsv',
]

RECOMMENDED = [
    ROOT / 'data' / 'precomputed' / 'model.joblib',
    ROOT / 'data' / 'precomputed' / 'metrics.csv',
    ROOT / 'data' / 'precomputed' / 'plots',
]

OPTIONAL = [
    ROOT / 'data' / 'demo_artifacts',
    ROOT / 'data' / 'validation_datasets' / 'blood_expr.tsv',
    ROOT / 'data' / 'validation_datasets' / 'cytometry_df.tsv',
]


def check(paths):
    missing = []
    for p in paths:
        if str(p).endswith('plots'):
            # directory check
            if not p.exists() or not p.is_dir():
                missing.append(str(p) + ' (dir)')
        else:
            if not p.exists():
                missing.append(str(p))
    return missing


def main():
    print('Checking Render readiness for folder:', ROOT)
    missing_req = check(REQUIRED)
    if missing_req:
        print('\nMISSING REQUIRED:')
        for m in missing_req:
            print(' -', m)
    else:
        print('\nAll required files present.')

    missing_rec = check(RECOMMENDED)
    if missing_rec:
        print('\nMISSING RECOMMENDED (precomputed artifacts):')
        for m in missing_rec:
            print(' -', m)
    else:
        print('\nRecommended precomputed artifacts present.')

    missing_opt = check(OPTIONAL)
    if missing_opt:
        print('\nOPTIONAL missing (fine to deploy, but some features may be limited):')
        for m in missing_opt:
            print(' -', m)
    else:
        print('\nOptional demo/validation files present.')

    print('\nDone.')


if __name__ == '__main__':
    main()
