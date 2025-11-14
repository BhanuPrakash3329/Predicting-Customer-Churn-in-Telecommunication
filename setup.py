from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning model to predict customer churn.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "flask",
        "joblib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)


