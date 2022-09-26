from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['suction_graspnet'],
	package_dir={'': 'network'}
)

setup(**setup_args)