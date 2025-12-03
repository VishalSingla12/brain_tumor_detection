# Brain Tumor Detection 

A complete Deep Learning project to detect brain tumors from MRI scans using a custom CNN trained from scratch. Includes **Grad-CAM** for explainability and a **Streamlit** web app for easy interaction.

##  Features
- **Custom CNN Architecture**: Trained from scratch on the Kaggle Brain Tumor MRI Dataset.
- **Grad-CAM Visualization**: See exactly where the model is looking.
- **Streamlit UI**: User-friendly interface for uploading images and viewing predictions.
- **Data Pipeline**: Automated download and splitting of dataset via `kagglehub`.
- **Reproducibility**: Seeded random generators and config-driven hyperparameters.

##  Setup

### Prerequisites
- Python 3.12 


### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd brain_tumor_detection
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `torch` on Python 3.12, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for the specific installation command for your system.*

##  Data Preparation
The dataset is downloaded automatically using `kagglehub`.

Run the data loader to download and prepare the data:
```bash
python data_loader.py
```
This will:
1. Download the "masoudnickparvar/brain-tumor-mri-dataset".
2. Create `data/train`, `data/val`, and `data/test` directories.
3. Split the data (70% Train, 15% Val, 15% Test).

##  Training
To train the model from scratch:

```bash
python train.py --epochs 25 --batch-size 32 --lr 1e-4
```
- **Checkpoints**: The best model (highest val accuracy) is saved to `best_model.pth`.
- **Logs**: Training metrics are saved to `logs/training_log.csv`.

##  Evaluation
To evaluate the trained model on the test set:

```bash
python eval.py
```
This will generate:
- Classification Report (`outputs/classification_report.csv`)
- Confusion Matrix (`outputs/confusion_matrix.png`)

##  Streamlit App
To launch the interactive web application:

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

##  Project Structure
```
brain_tumor_detection/
├── data/               # Dataset (created after running data_loader.py)
├── logs/               # Training logs
├── outputs/            # Evaluation outputs
├── notebooks/          # Jupyter notebooks
├── app.py              # Streamlit application
├── config.py           # Configuration & Hyperparameters
├── data_loader.py      # Data download & preparation
├── eval.py             # Evaluation script
├── gradcam.py          # Grad-CAM implementation
├── model.py            # CNN Architecture
├── train.py            # Training script
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

##  Troubleshooting
- **Python 3.12 Issues**: If some libraries are not yet compatible with 3.12, try creating a Python 3.11 environment:
  ```bash
  python3.11 -m venv venv
  ```
- **CUDA/GPU**: The code automatically detects CUDA. Ensure you have the correct NVIDIA drivers and PyTorch CUDA version installed if you want to use GPU.
