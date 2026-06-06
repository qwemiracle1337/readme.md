from pathlib import Path
import joblib

def test_model_pkl_exists():
    root = Path(__file__).parent.parent
    artifacts_dir = root / 'artifacts'
    pkl_files = list(artifacts_dir.glob('*.pkl'))
    assert len(pkl_files) > 0
    for pkl in pkl_files:
        model = joblib.load(pkl)
        assert hasattr(model, 'predict'), "Загруженный объект не модель"