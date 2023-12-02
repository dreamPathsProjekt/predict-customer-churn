#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="customer_churn",
    version="0.0.1",
    description="Project Predict Customer Churn of ML DevOps Engineer Nanodegree Udacity",
    author="dreamPathsProjekt",
    url="https://github.com/dreamPathsProjekt/predict-customer-churn",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn==0.24.1",
        "shap==0.40.0",
        "joblib==1.0.1",
        "pandas==1.2.4",
        "numpy==1.20.1",
        "matplotlib==3.3.4",
        "seaborn==0.11.2",
    ],
    extras_require={
        "develop": [
            "pylint==2.7.4",
            "autopep8==1.5.6",
            "pytest==7.4.3",
            "dvc==3.30.3",
        ]
    },
    entry_points={
        "console_scripts": []
    },
)
