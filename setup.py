# setup.py
from setuptools import setup, find_packages

setup(
    name='solving-pde-problems',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],  # Leave this empty if using environment.yml
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
            # e.g., 'my_project=module1:main_function'
        ],
    },
)