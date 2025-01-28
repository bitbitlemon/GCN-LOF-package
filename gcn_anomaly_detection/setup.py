from setuptools import setup, find_packages

setup(
    name="gcn_anomaly_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torch-geometric",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
    description="A Python package for anomaly detection using Graph Convolutional Networks (GCN).",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/gcn_anomaly_detection",
)