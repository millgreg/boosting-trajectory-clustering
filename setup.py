from setuptools import setup, find_packages

setup(
    name='btc-clustering',
    version='0.2.0',
    description='Boosting Trajectory Clustering (BTC)',
    author='Miller et al.',
    packages=find_packages(include=['btc', 'btc.*', 'baselines', 'baselines.*']),
    install_requires=[
        'numpy>=1.26.0',
        'scipy>=1.11.0',
        'scikit-learn>=1.3.0',
    ],
    extras_require={
        'xgboost': ['xgboost>=2.0.0'],
        'lightgbm': ['lightgbm>=4.0.0'],
    },
    python_requires='>=3.9',
)
