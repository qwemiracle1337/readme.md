import sys
from pathlib import Path
import re

def test_artifacts_exist():
    root = Path(__file__).parent.parent
    artifacts_dir = root / 'artifacts'
    expected_files = ['recall_score.txt', 'classification_report.txt']
    for fname in expected_files:
        file_path = artifacts_dir / fname
        assert file_path.exists(), f"Файл {fname} не существует"
        assert file_path.stat().st_size > 0, f"Файл {fname} пуст"