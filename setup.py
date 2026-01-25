from setuptools import setup, find_packages

setup(
    name="cfd",
    version="1.0.0",
    description="CFD Heat Transfer Solver Package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.8.0",
    ],
    python_requires=">=3.7",
)


