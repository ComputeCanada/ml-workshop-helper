import os.path as osp
from gzip import decompress
from io import BytesIO

import pandas as pd


def _get_file_dir() -> str:
    return osp.dirname(osp.realpath(__file__))


def load() -> pd.DataFrame:
    with open(osp.join(_get_file_dir(), './conductors.csv.gz'), 'rb') as f:
        gz_bytes = f.read()

    csv_file = BytesIO(decompress(gz_bytes))
    out = pd.read_csv(csv_file, index_col='id')

    Y = out[['formation_energy_ev_natom', 'bandgap_energy_ev']]
    X = out.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)

    return X, Y


def __test():
    X, Y = load()
    print(X.head())
    print(Y.head())


if __name__ == '__main__':
    __test()
