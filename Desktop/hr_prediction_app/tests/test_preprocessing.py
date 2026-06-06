import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_processed_files_have_no_nan():
    root = Path(__file__).parent.parent
    processed_folder = root / 'data' / 'processed'
    assert processed_folder.exists()
    csv_files = list(processed_folder.glob('*.csv'))
    assert len(csv_files) > 0
    for file in csv_files:
        df = pd.read_csv(file)
        assert df.isnull().sum().sum() == 0, f"NaN в {file.name}"