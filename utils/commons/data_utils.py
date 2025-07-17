import os
from pathlib import Path
import csv

def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path

def load_samples_from_tsv(tsv_path):
    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        print(f"warning: empty manifest: {tsv_path}")
        return []
    return samples

def load_dict_from_tsv(tsv_path, key):
    samples = load_samples_from_tsv(tsv_path)
    samples = {sample[key]: sample for sample in samples}
    return samples

