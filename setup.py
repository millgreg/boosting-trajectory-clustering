from setuptools import setup, find_packages

setup(
    name='btc-clustering',
    version='0.1.0',
    description='Boosting Trajectory Clustering',
    author='Miller et al.',
    packages=find_packages(include=['btc', 'btc.*']),
    install_requires=[
        'numpy',
        'scikit-learn',
        'xgboost',
        'lightgbm'
    ],
    python_requires='>=3.8',
)
