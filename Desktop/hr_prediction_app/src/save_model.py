import pickle
import os


def save_model(model, name: str = "LogisticRegression"):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(root_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, f'{name}.pkl'), 'wb') as f:
        pickle.dump(model, f)