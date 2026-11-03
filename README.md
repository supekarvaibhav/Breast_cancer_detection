# BreastGuard AI – Breast Cancer Detection Platform

A full-stack AI platform for early breast cancer detection using a **CNN + SVM Hybrid Model** trained on the **BreakHis** histopathology dataset, paired with a Flask web application where patients and doctors can interact in real time.

---

## Architecture Overview

```
Input Image (224×224 RGB)
       │
EfficientNetB0 (ImageNet pre-trained, fine-tuned)
       │
GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout
       │
       256-D Feature Vector
       │
SVM Classifier (RBF, C=10)   ←─ trained on extracted features
       │
   Benign / Malignant  +  Confidence Score
```

**Target Accuracy: ≥ 90% on BreakHis test set**

---

## Project Structure

```
breast_cancer_detection/
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── ml/
│   │   ├── model.py           # CNN+SVM hybrid architecture
│   │   └── saved_models/      # Trained model files (after training)
│   ├── models/
│   │   ├── user.py            # User, PatientProfile, DoctorProfile
│   │   ├── diagnosis.py       # Diagnosis model
│   │   └── consultation.py    # Consultation, ChatMessage
│   ├── routes/
│   │   ├── auth.py            # Register / Login / Logout
│   │   ├── patient.py         # Upload, Result, History
│   │   ├── doctor.py          # Dashboard, Accept, Complete
│   │   ├── main.py            # Home, About
│   │   └── chat.py            # SocketIO real-time chat
│   ├── static/
│   │   ├── css/style.css
│   │   └── uploads/           # Patient uploaded images
│   └── templates/             # Jinja2 HTML templates
├── train.py                   # Model training script
├── run.py                     # Flask application entry point
├── config.py                  # Configuration classes
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone / Navigate to project
```bash
cd C:\breast_cancer_detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or: source venv/bin/activate  (Linux/Mac)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the database
For hosted use, create a Neon project and set `DATABASE_URL` to its pooled Postgres connection string. For local SQLite development, use:
```bash
set DATABASE_URL=sqlite:///breast_cancer.db   # Windows
# or: export DATABASE_URL=sqlite:///breast_cancer.db  (Linux/Mac)
```

### 5. Download BreakHis Dataset
Download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

Extract so the structure is:
```
BreaKHis_v1/
  histology_slides/
    breast/
      benign/SOB/adenosis/ ...
      malignant/SOB/ductal_carcinoma/ ...
```

### 6. Train the Model
```bash
python train.py --dataset /path/to/BreaKHis_v1 --magnification ALL --epochs 30
```

This will:
- Phase 1: Pre-train EfficientNetB0 feature extractor (frozen base)
- Phase 2: Fine-tune top 30 layers of EfficientNetB0
- Phase 3: Train SVM on extracted deep features
- Save models to `app/ml/saved_models/`
- Output training curves and confusion matrix

### 7. Initialize Database & Run
```bash
python run.py
```

The first startup can take 20-60 seconds while TensorFlow and the saved model load.

Visit `http://localhost:5001`

---

## Deploy on Render

The root `render.yaml` Blueprint defines a Python web service that connects to Neon Postgres and Neon Object Storage. The web service uses a 2 GB instance because TensorFlow loads the CNN at startup. Patient images are kept in a private S3-compatible Neon bucket; the database stores only each image's object key.

1. Create a Neon project in `aws-ap-southeast-1` (Singapore), which is supported by Neon Object Storage.
2. Install Node.js 20.19 or later, then run `npm install` to install the config package. Install the Neon CLI with `npm install --global neon`, authenticate with `neon auth`, link this repository to the Neon project with `neon link`, and run `neon deploy` from the repository root to create the private bucket declared in `neon.ts`.
3. Run `neon env pull` and copy `DATABASE_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`, and `AWS_REGION` into the environment-variable prompts when creating the Render Blueprint. Use the pooled Neon `DATABASE_URL` for the app.
4. Push this repository to a Git provider connected to Render. Include the files in `app/ml/saved_models/`; the CNN, SVM, and scaler artifacts are required for predictions.
5. In Render, choose **New + Blueprint**, select the repository, and review the web service plan before applying it.
6. Render builds and deploys the service. The `/health` endpoint checks database connectivity and is used as the deploy health check. Open the service URL shown in the Render dashboard.

The Blueprint generates `SECRET_KEY`; Neon supplies the database and private-storage credentials. During local development, when the Neon storage variables are absent, uploaded images are saved under `app/static/uploads`. In production, startup fails if the required Neon Object Storage variables are missing.

If you deploy the app without the Blueprint, use this start command:

The currently bundled model artifacts are marked as a demo model in `metrics.json`. Replace them with validated training outputs before using predictions beyond a demonstration; this app is not a substitute for clinical diagnosis.

```bash
gunicorn --worker-class gthread --workers 1 --threads 4 --bind 0.0.0.0:$PORT run:app
```

---

## Features

### For Patients
- Register / Login
- Upload histopathology slide images (PNG, JPG, TIF)
- Instant AI analysis (Benign / Malignant + confidence %)
- Risk level assessment (Low → Very High)
- View full diagnosis history
- Request real-time consultation with available doctors

### For Doctors
- Register / Login as doctor (add specialization, hospital, license)
- Dashboard showing pending consultation requests
- Review AI diagnosis with image and probability scores
- **AI as decision-support tool** — see probabilities before accepting
- Accept consultations and chat in real time
- Add clinical notes and complete consultations
- Toggle availability on/off

### Real-time Chat
- Socket.IO powered instant messaging
- Diagnosis context shown in chat header
- Auto-scroll to latest messages
- Enter key to send

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | EfficientNetB0 (ImageNet) |
| Feature Dim | 256-D after custom dense head |
| SVM Kernel | RBF |
| SVM C | 10.0 |
| SVM Gamma | scale |
| Image Size | 224 × 224 |
| Augmentation | Rotation, Flip, Zoom, Shift |
| Dataset Split | 70% Train / 15% Val / 15% Test |
| Classes | Binary (Benign=0, Malignant=1) |

### Training Strategy
1. **Phase 1** – Frozen EfficientNetB0, train custom head only
2. **Phase 2** – Unfreeze top 30 layers, fine-tune with 10× lower LR
3. **Phase 3** – Extract 256-D features → train SVM (RBF, C=10)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 3.0, Flask-SQLAlchemy, Flask-Login |
| Real-time | Flask-SocketIO (Socket.IO 4.x) |
| Database | PostgreSQL (SQLAlchemy ORM) |
| ML | TensorFlow/Keras 2.17 + scikit-learn |
| CNN Base | EfficientNetB0 (ImageNet) |
| Classifier | SVM (RBF kernel) |
| Frontend | Bootstrap 5.3 + Bootstrap Icons |
| Fonts | Google Fonts – Inter |

---

## Medical Disclaimer

> This tool is for **educational and research purposes only**. It does **not** replace professional medical diagnosis or treatment. Always consult a qualified physician for medical advice.

---

## License

MIT License – Free for educational use.
