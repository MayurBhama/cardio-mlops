import os
from pathlib import Path

project_name = "mlops-project"

# Full Directory & File Structure
list_of_files = [
    # ---------------- CONFIGS ----------------
    "configs/model_config.yaml",
    "configs/data_config.yaml",

    # ---------------- DATA ----------------
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",

    # ---------------- MODELS ----------------
    "models/trained_models/.gitkeep",
    "models/model_artifacts/.gitkeep",

    # ---------------- NOTEBOOKS ----------------
    "notebooks/exploration.ipynb",
    "notebooks/experimentation.ipynb",

    # ---------------- SRC PACKAGE ----------------
    "src/__init__.py",

    # INGESTION
    "src/data_ingestion/ingestion.py",

    # PROCESSING
    "src/data_processing/cleaning.py",

    # FEATURE ENGINEERING
    "src/feature_engineering/features.py",

    # TRAINING
    "src/model_training/train.py",

    # EVALUATION
    "src/model_evaluation/evaluate.py",

    # DEPLOYMENT
    "src/deployment/serve.py",

    # ---------------- TESTS ----------------
    "tests/data_tests/__init__.py",
    "tests/data_tests/test_data_quality.py",
    "tests/model_tests/__init__.py",
    "tests/model_tests/test_model.py",

    # ---------------- DEPLOYMENT ----------------
    "deployment/Dockerfile",
    "deployment/kubernetes/deployment.yaml",

    # ---------------- SCRIPTS ----------------
    "scripts/run_pipeline.sh",
    "scripts/train_model.sh",

    # ---------------- ROOT FILES ----------------
    ".gitattributes",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "dvc.yaml"
]


def create_structure():
    for filepath in list_of_files:
        filepath = Path(project_name) / filepath
        filedir, filename = os.path.split(filepath)

        # create folders
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)

        # create empty/placeholder files
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                if filename.endswith(".py"):
                    f.write("# Auto-generated file\n")
                elif filename.endswith(".sh"):
                    f.write("#!/bin/bash\n")
                elif filename.endswith(".yaml") or filename.endswith(".yml"):
                    f.write("# config\n")
                else:
                    f.write("")
            print(f" Created: {filepath}")
        else:
            print(f" Exists:  {filepath}")


if __name__ == "__main__":
    print(f"\n Initializing project structure for: {project_name}\n")
    create_structure()
    print(f"\n Project structure created successfully in folder: {project_name}/\n")