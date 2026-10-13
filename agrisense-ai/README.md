# AgriSense AI — Weather-Aware Plant Doctor 🌿

An educational AI application that screens uploaded plant-leaf images using a transfer-learning classifier, presents crop-care guidance, and combines live weather data with general scouting reminders.

## Features

- Leaf image classification using MobileNetV2 transfer learning
- Model score and top-three class outputs
- Class-specific, cautious crop-care guidance
- Current weather and three-day forecast from Open-Meteo
- General weather-aware scouting notes
- Streamlit web interface
- Training script for a folder-organized plant disease dataset

> **Important limitations:** This is a student prototype, not a validated diagnostic or agricultural advisory tool. Predictions may be wrong, and the weather heuristics are not a calibrated disease-risk model. Confirm diagnoses and any treatment decision with local agricultural experts. Never apply chemicals based only on this application.

## Technology stack

Python, TensorFlow/Keras, MobileNetV2, Streamlit, Pillow, NumPy, Open-Meteo API.

## Folder structure

```text
agrisense-ai/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── (created after training; model files are excluded from Git)
```

## Requirements

- Python 3.10 or 3.11 recommended for compatibility with TensorFlow
- VS Code and its Python extension
- Internet connection for dependency installation and Open-Meteo weather
- PlantVillage dataset or another properly licensed, labeled dataset

## Run in VS Code on Windows

### 1. Download and extract

Download the ZIP from the project handoff, extract it, and open the extracted folder in VS Code.

### 2. Create a virtual environment

Open **Terminal → New Terminal** and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not installed, install it from https://www.python.org/downloads/ or use an installed Python version supported by the selected TensorFlow release.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation can take a while and needs a compatible Python/platform combination.

### 4. Download a dataset

A commonly used starting point is the PlantVillage plant disease dataset:
https://www.kaggle.com/datasets/emmarex/plantdisease

Review its license and terms before redistribution. Download and extract the images locally. The training script expects a directory with one folder per class, for example:

```text
PlantVillage/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
├── Potato___Early_blight/
└── ...
```

Folder names become the model's class labels. If the extracted dataset has an extra parent directory, pass the directory that directly contains the class folders.

### 5. Train the model

From the project root, run:

```powershell
python train_model.py --data_dir "C:\path\to\PlantVillage" --epochs 5
```

Replace the example path with your actual dataset folder. Training time depends on your computer and dataset. A CPU can take a while. The script creates:

```text
models/
├── plant_disease_model.keras
└── class_names.json
```

The app will not perform predictions until both files exist. Inspect the training and validation metrics; a validation split alone does not prove real-world performance.

### 6. Launch the app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

Upload a leaf photo, click **Analyze leaf**, enter latitude and longitude, and click **Get current weather and forecast**. Weather coordinates default to an example location and should be changed to your chosen location.

## GitHub upload steps

### Option A — VS Code

1. Sign in at https://github.com/ and create a new repository named `agrisense-ai`.
2. Do not initialize the remote repository with a README.
3. Open the project folder in VS Code.
4. Click **Source Control → Initialize Repository**.
5. Stage the source files, commit with a message such as `Initial commit: AgriSense AI plant doctor`, then select **Publish Branch**.
6. Sign in and publish to the repository you created.

### Option B — Git commands

Run these from the project folder after creating an empty GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit: AgriSense AI plant doctor"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/agrisense-ai.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Install Git from https://git-scm.com/downloads if needed.

### Model files and GitHub

The trained `.keras` model is excluded from Git because it may be large. The source repository can contain the training code and instructions, but a fresh clone needs you to train the model or download a model artifact from a trusted, documented source. If you distribute a trained model, check the dataset/model licenses and use Git LFS or an approved release/artifact host when appropriate.

Do not upload private credentials, API keys, or personal data.

## Suggested improvements for a stronger final-year project

1. Evaluate on a held-out dataset and real field photos; report per-class precision, recall, F1, and a confusion matrix.
2. Add Grad-CAM visual explanations.
3. Add multilingual Telugu/English crop-care explanations grounded in vetted agricultural sources.
4. Improve weather recommendations with crop-specific, locally validated rules and uncertainty reporting.
5. Add tests, model versioning, and a model card documenting dataset, limitations, and evaluation.

## Resume description

**AgriSense AI — Weather-Aware Plant Doctor | Python, TensorFlow, MobileNetV2, Streamlit, Open-Meteo**

- Built a plant-leaf image screening application using transfer learning with MobileNetV2.
- Integrated live weather and forecast data to surface general weather-aware crop scouting reminders.
- Developed an interactive dashboard with class scores, crop-care guidance, and a documented training pipeline.

Only claim measured model performance after training and evaluating the model yourself.
