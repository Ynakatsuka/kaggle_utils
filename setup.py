from setuptools import setup, find_packages


setup(
    name='kaggle_utils',
    version='0.0.1',
    install_requires=_requires_from_file('requirements.txt'),
    description='utility scripts for kaggle',
    author='Yuki Nakatsuka',
)
