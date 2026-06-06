import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    def __init__(self, raw_data_path: Optional[Path] = None):
        if raw_data_path is None:
            root_dir = Path(__file__).parent.parent
            self.raw_data_path = root_dir / 'data' / 'raw' / 'hr_data.csv'
        else:
            self.raw_data_path = raw_data_path

    def load_data(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.raw_data_path}. "
                "Please place the CSV file in data/raw/hr_data.csv"
            )
        df = pd.read_csv(self.raw_data_path)
        if 'Attrition' in df.columns:
            df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
        return df


def load_data(raw_folder: str = 'raw') -> pd.DataFrame:
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / 'data' / raw_folder / 'hr_data.csv'
    loader = DataLoader(raw_data_path=data_path)
    return loader.load_data()