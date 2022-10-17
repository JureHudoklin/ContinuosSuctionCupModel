from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['suction_graspnet', 'config', 'point_cloud_reader'],
	package_dir={'': 'network', '': 'network', '': 'scene_render'}
)

setup(**setup_args)