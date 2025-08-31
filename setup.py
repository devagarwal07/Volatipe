from setuptools import setup, find_packages

setup(
    name="volatility-india",
    version="0.1.0",
    description="MSAE-India Volatility Prediction System",
    author="MSAE-India Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # kept minimal; full list in requirements.txt
        "pandas",
        "numpy",
    ],
    python_requires=">=3.9",
)
