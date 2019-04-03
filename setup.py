import numpy.distutils.misc_util as ndist_misc
from setuptools import Extension, setup

# semver with automatic minor bumps keyed to unix time
__version__ = '1.0.1554323447'


setup(
    name="ml_helper",
    version=__version__,
    packages=["ml_helper"],
    include_package_data=True,
)
