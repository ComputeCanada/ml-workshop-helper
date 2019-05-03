import numpy.distutils.misc_util as ndist_misc
from setuptools import Extension, setup

# semver with automatic minor bumps keyed to unix time
__version__ = '1.0.1556910073'

setup(
    name="ml_helper",
    version=__version__,
    packages=["ml_helper"],
    data_files=[
        (
            'data', [
                'ml_helper/data/conductors_test.csv.gz',
                'ml_helper/data/conductors_train.csv.gz'
            ]
        ),
    ],
    include_package_data=True,
)
