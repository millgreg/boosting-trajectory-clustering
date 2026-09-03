from setuptools import setup, find_packages

setup(
    name='rtc-clustering',
    version='0.1.0',
    description='Residual Trajectory Clustering',
    author='Miller et al.',
    packages=find_packages(include=['rtc', 'rtc.*']),
    install_requires=[
        'numpy',
        'scikit-learn',
        'xgboost',
        'lightgbm'
    ],
    python_requires='>=3.8',
)
