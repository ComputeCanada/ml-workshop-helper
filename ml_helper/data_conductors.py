import os.path as osp
from gzip import decompress
from io import BytesIO

import pandas as pd


def _get_file_dir() -> str:
    return osp.dirname(osp.realpath(__file__))


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:

    Y = df[['formation_energy_ev_natom', 'bandgap_energy_ev']]
    X = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)

    return X, Y


def _load(fn) -> pd.DataFrame:
    with open(osp.join(_get_file_dir(), fn), 'rb') as f:
        gz_bytes = f.read()

    csv_file = BytesIO(decompress(gz_bytes))
    out = pd.read_csv(csv_file, index_col='id')

    return _clean_df(out)


def load():
    return _load('./conductors_train.csv.gz')


def _load_test():
    return _load('./conductors_test.csv.gz')


def __test():
    X, Y = load()
    print(X.head())
    print(Y.head())


if __name__ == '__main__':
    __test()
