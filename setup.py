# Setup script
from setuptools import setup, find_packages

setup(
    name="rag-rac-naics",
    version="0.1.0",
    description="RAG with Retrieval-Augmented Classification and NAICS Classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.32.5",
        "python-dotenv>=1.1.1",
        "openai>=1.3.0",
        "google-generativeai>=0.3.2",
        "chromadb>=0.4.15",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.2",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.9.1",
            "flake8>=6.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rag-rac-naics=rag_rac_naics.main:main",
        ],
    },
)
