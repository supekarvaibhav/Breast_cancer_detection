# BreastGuard AI – Complete Project Explanation
## Breast Cancer Detection Using CNN + SVM Hybrid Model

---

## 1. PROJECT OVERVIEW

BreastGuard AI is a full-stack web application for early breast cancer detection. It uses a
hybrid deep learning approach combining a Convolutional Neural Network (CNN) with a Support
Vector Machine (SVM) classifier to analyze histopathology slide images and classify them as
either **Benign** (non-cancerous) or **Malignant** (cancerous).

The platform also provides:
- Role-based user system (Patient and Doctor)
- Real-time chat between patients and doctors via WebSockets
- Diagnosis history tracking
- Doctor consultation management

---

## 2. TECHNOLOGIES USED

### 2.1 Backend (Server-Side)

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.x | Core programming language |
| **Flask** | 3.0.3 | Lightweight web framework for building the server |
| **Flask-SQLAlchemy** | 3.1.1 | ORM (Object Relational Mapper) for database operations |
| **Flask-Login** | 0.6.3 | User session management and authentication |
| **Flask-SocketIO** | 5.3.6 | Real-time bidirectional communication (WebSockets) |
| **Flask-Migrate** | 4.0.7 | Database schema migration management |
| **Flask-Mail** | 0.10.0 | Email notification support |
| **Flask-WTF** | 1.2.1 | Form handling and CSRF protection |
| **Werkzeug** | 3.0.3 | WSGI utilities (password hashing, file handling) |
| **Eventlet** | 0.37.0 | Async networking library for SocketIO |

### 2.2 Machine Learning & AI

| Technology | Version | Purpose |
|------------|---------|---------|
| **TensorFlow** | 2.17.0 | Deep learning framework for building and running the CNN |
| **Keras** | 3.5.0 | High-level neural network API (integrated with TensorFlow) |
| **EfficientNetB0** | (ImageNet) | Pre-trained CNN model used as the feature extractor base |
| **scikit-learn** | 1.5.2 | SVM classifier, StandardScaler, evaluation metrics |
| **OpenCV** | 4.10.0 | Image reading, resizing, color conversion |
| **Pillow (PIL)** | 10.4.0 | Image manipulation and preprocessing |
| **NumPy** | 1.26.4 | Numerical array operations |
| **SciPy** | 1.14.1 | Scientific computing utilities |
| **joblib** | 1.4.2 | Saving/loading SVM model and scaler to disk |
| **imbalanced-learn** | 0.12.3 | Handling class imbalance in training data |

### 2.3 Data Visualization

| Technology | Version | Purpose |
|------------|---------|---------|
| **Matplotlib** | 3.9.2 | Plotting training curves (accuracy, loss) |
| **Seaborn** | 0.13.2 | Confusion matrix heatmap visualization |
| **Pandas** | 2.2.3 | Data manipulation and analysis |

### 2.4 Database

| Technology | Purpose |
|------------|---------|
| **SQLite** | Lightweight file-based relational database |
| **SQLAlchemy** | ORM layer for Python-to-SQL mapping |

### 2.5 Frontend (Client-Side)

| Technology | Purpose |
|------------|---------|
| **HTML5** | Page structure and semantic markup |
| **CSS3** | Styling with custom properties, gradients, animations |
| **JavaScript** | Client-side interactivity and SocketIO client |
| **Bootstrap 5.3** | Responsive UI framework (grid, components, utilities) |
| **Bootstrap Icons** | Icon library for UI elements |
| **Google Fonts (Inter)** | Modern typography |
| **Socket.IO Client** | Real-time messaging on the frontend |
| **Jinja2** | Server-side HTML templating engine (built into Flask) |

---

## 3. AI/ML MODEL ARCHITECTURE (CNN + SVM Hybrid)

### 3.1 Why a Hybrid Model?

The project uses a two-stage hybrid approach:
1. **CNN (EfficientNetB0)** acts as an automatic feature extractor — it learns to convert a
   raw 224×224 pixel image into a compact 256-dimensional feature vector.
2. **SVM (Support Vector Machine)** takes those 256 features and performs the final
   Benign vs. Malignant classification.

This hybrid approach often outperforms using CNN alone because:
- SVMs excel at binary classification with clear margin separation
- CNN features are richer than hand-crafted features
- The combination reduces overfitting on small medical datasets

### 3.2 Model Pipeline

```
Input Image (224 × 224 × 3 RGB)
        │
        ▼
┌─────────────────────────────┐
│  EfficientNetB0 (ImageNet)  │  ← Pre-trained on 1.2M images
│  (Frozen or Fine-tuned)     │  ← Transfer learning
└─────────────────────────────┘
        │
        ▼
  GlobalAveragePooling2D          ← Reduces spatial dims to 1D
        │
        ▼
  Dense(512, ReLU)                ← Fully connected layer
  BatchNormalization              ← Normalizes activations
  Dropout(0.4)                    ← Prevents overfitting
        │
        ▼
  Dense(256, ReLU)                ← Feature output layer ("feature_out")
        │
        ▼
┌─────────────────────────────┐
│  256-Dimensional Feature    │
│  Vector                     │
└─────────────────────────────┘
        │
        ▼
  StandardScaler                  ← Normalizes features to zero mean, unit variance
        │
        ▼
┌─────────────────────────────┐
│  SVM Classifier             │
│  Kernel: RBF                │
│  C: 10.0                    │
│  Gamma: scale               │
│  Probability: enabled       │
└─────────────────────────────┘
        │
        ▼
  Output: Benign (0) or Malignant (1) + Confidence %
```

### 3.3 Training Strategy (3 Phases)

**Phase 1 – CNN Pre-training (30 epochs)**
- EfficientNetB0 base is FROZEN (weights don't change)
- Only the custom head (Dense layers) is trained
- Uses binary cross-entropy loss with Adam optimizer (LR = 0.0001)
- Data augmentation: rotation, flipping, zooming, shifting

**Phase 2 – CNN Fine-tuning (15 epochs)**
- Top 30 layers of EfficientNetB0 are UNFROZEN
- Trained with 10× lower learning rate (LR = 0.00001)
- Allows the CNN to adapt its learned features to breast tissue patterns

**Phase 3 – SVM Training**
- The fine-tuned CNN extracts 256-D features from all images
- Features are scaled with StandardScaler
- SVM (RBF kernel, C=10) is trained on the scaled features
- No epochs — SVM training is a single optimization pass

### 3.4 Dataset: BreakHis

The model is designed for the **Breast Cancer Histopathological Database (BreakHis)**:
- 7,909 histopathology images at 4 magnifications (40X, 100X, 200X, 400X)
- 2 classes: Benign (4 subtypes) and Malignant (4 subtypes)
- Benign subtypes: Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma
- Malignant subtypes: Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma

### 3.5 Model Parameters

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 (RGB) |
| Base Model | EfficientNetB0 (ImageNet pre-trained) |
| Feature Dimension | 256 |
| SVM Kernel | RBF (Radial Basis Function) |
| SVM Regularization (C) | 10.0 |
| SVM Gamma | scale |
| Dataset Split | 70% Train / 15% Validation / 15% Test |
| Target Accuracy | ≥ 90% on BreakHis |

---

## 4. PROJECT FILE STRUCTURE (Detailed)

```
breast_cancer_detection/
│
├── run.py                         # Application entry point
├── config.py                      # Configuration classes (dev/prod)
├── train.py                       # Full model training script
├── create_demo_model.py           # Creates placeholder demo model
├── requirements.txt               # Python package dependencies
├── setup.bat                      # Windows first-time setup script
├── .env.example                   # Environment variable template
├── breast_cancer.db               # SQLite database file
├── README.md                      # Project documentation
│
├── app/                           # Flask application package
│   ├── __init__.py                # App factory + extension init
│   │
│   ├── ml/                        # Machine Learning module
│   │   ├── __init__.py
│   │   ├── model.py               # CNN+SVM architecture + inference
│   │   └── saved_models/          # Trained model files
│   │       ├── cnn_feature_extractor.keras   (20 MB)
│   │       ├── svm_classifier.joblib         (861 KB)
│   │       └── feature_scaler.joblib         (6 KB)
│   │
│   ├── models/                    # SQLAlchemy database models
│   │   ├── __init__.py
│   │   ├── user.py                # User, PatientProfile, DoctorProfile
│   │   ├── diagnosis.py           # Diagnosis records
│   │   └── consultation.py        # Consultation + ChatMessage
│   │
│   ├── routes/                    # Flask blueprints (URL handlers)
│   │   ├── __init__.py
│   │   ├── auth.py                # Register, Login, Logout
│   │   ├── patient.py             # Upload, Result, History, Request Consultation
│   │   ├── doctor.py              # Dashboard, Accept, Complete, Toggle Availability
│   │   ├── main.py                # Home page, About page, Awareness page
│   │   └── chat.py                # SocketIO real-time chat
│   │
│   ├── templates/                 # Jinja2 HTML templates
│   │   ├── base.html              # Master layout template
│   │   ├── index.html             # Homepage
│   │   ├── about.html             # About page
│   │   ├── awareness.html         # Breast cancer awareness page
│   │   ├── auth/
│   │   │   ├── login.html         # Login form
│   │   │   └── register.html      # Registration form (patient/doctor)
│   │   ├── patient/
│   │   │   ├── dashboard.html     # Patient dashboard with stats
│   │   │   ├── upload.html        # Image upload for analysis
│   │   │   ├── result.html        # AI prediction results
│   │   │   └── history.html       # Diagnosis history with pagination
│   │   ├── doctor/
│   │   │   ├── dashboard.html     # Doctor dashboard (pending/active)
│   │   │   ├── patients.html      # List of completed consultations
│   │   │   └── view_diagnosis.html# View specific diagnosis details
│   │   └── chat/
│   │       └── room.html          # Real-time chat room
│   │
│   └── static/                    # Static assets
│       ├── css/style.css          # Custom CSS styles
│       ├── js/                    # JavaScript files
│       └── uploads/               # Patient uploaded images
│
└── venv/                          # Python virtual environment
```

---

## 5. DETAILED FILE EXPLANATIONS

### 5.1 run.py (Entry Point)
- Creates the Flask app using the factory pattern
- Initializes the database tables on startup (db.create_all)
- Runs the app with SocketIO support on host 0.0.0.0, port 5000
- Provides a shell context for debugging (access models in Flask shell)
- Provides a CLI command `flask init-db` to initialize the database

### 5.2 config.py (Configuration)
- `Config` base class with all settings:
  - SECRET_KEY for session security
  - SQLite database URI
  - Upload folder path and allowed file extensions (png, jpg, jpeg, tif, bmp)
  - Max upload size: 16 MB
  - Model file paths (CNN, SVM, Scaler)
  - Image dimensions: 224×224×3
  - Training hyperparameters (batch=32, epochs=30, lr=0.0001)
  - Session lifetime: 12 hours
  - Mail server configuration
- `DevelopmentConfig` adds DEBUG=True
- `ProductionConfig` has DEBUG=False

### 5.3 app/__init__.py (App Factory)
- Initializes Flask extensions: SQLAlchemy, LoginManager, SocketIO, Migrate
- Registers 5 blueprints with URL prefixes:
  - `auth_bp` → /register, /login, /logout
  - `patient_bp` → /patient/*
  - `doctor_bp` → /doctor/*
  - `main_bp` → / (root)
  - `chat_bp` → /chat/*
- Creates upload and model directories
- Pre-loads the ML model into `app.ml_model` on startup

### 5.4 app/ml/model.py (AI Model)
This is the core ML file containing:

**build_feature_extractor()** — Constructs the CNN:
- EfficientNetB0 base (ImageNet weights, frozen)
- GlobalAveragePooling2D → Dense(512) → BatchNorm → Dropout(0.4) → Dense(256)
- Outputs a 256-dimensional feature vector

**build_full_cnn_classifier()** — Adds a classification head:
- Takes the feature extractor output
- Adds Dropout(0.3) → Dense(1, sigmoid) for binary classification
- Used during training only; replaced by SVM in production

**SVMClassifier** — Wrapper class:
- Wraps sklearn's SVC with StandardScaler
- Methods: fit(), predict(), predict_proba(), save(), load()

**BreastCancerHybridModel** — Inference class used by Flask:
- Loads saved CNN + SVM + Scaler from disk
- _preprocess(): Resizes image to 224×224, keeps pixel range [0,255]
- predict_from_array(): Full pipeline — preprocess → CNN features → SVM predict
- Returns: prediction label, confidence %, probabilities, risk level, recommendations

**get_model()** — Singleton loader:
- Loads model once and caches it for all requests
- Gracefully handles missing model files

**_risk_level()** — Maps malignancy probability to risk:
- < 30% → Low
- 30-60% → Moderate
- 60-80% → High
- > 80% → Very High

**_recommendations()** — Generates medical recommendations based on prediction

### 5.5 app/models/user.py (User Models)

**User** — Main user table:
- Fields: email, username, password_hash, role (patient/doctor), full_name, phone, avatar
- Password hashing via Werkzeug (generate_password_hash / check_password_hash)
- One-to-one relationships to PatientProfile or DoctorProfile

**PatientProfile** — Patient-specific data:
- Fields: dob, gender, address, medical_history
- Has many Diagnosis records

**DoctorProfile** — Doctor-specific data:
- Fields: specialization, hospital, license_number, years_experience, bio, is_available
- Has many Consultation records

### 5.6 app/models/diagnosis.py
**Diagnosis** — Stores each AI analysis:
- Links to patient_profile (who uploaded)
- Stores: image_filename, prediction, is_malignant, confidence, prob_benign, prob_malignant
- Stores: risk_level, doctor_notes, reviewed_by_id
- Has one optional Consultation

### 5.7 app/models/consultation.py
**Consultation** — Doctor-patient consultation:
- Links diagnosis, patient, and doctor
- Status flow: requested → accepted → in_progress → completed (or cancelled)
- Tracks: scheduled_at, started_at, ended_at, notes
- Has many ChatMessage records

**ChatMessage** — Individual chat messages:
- Fields: sender_id, content, timestamp, is_read
- to_dict() for JSON serialization to SocketIO

### 5.8 app/routes/auth.py (Authentication)
- **POST /register** — Creates user with role-specific profile:
  - Validates email, username (≥3 chars), password (≥6 chars), confirmation match
  - Checks for duplicate email/username
  - Creates PatientProfile (with age, gender, medical history) or DoctorProfile (with specialization, hospital, license)
  - Auto-logs in after registration
- **POST /login** — Authenticates by email OR username, supports "remember me"
- **GET /logout** — Ends session

### 5.9 app/routes/patient.py (Patient Features)
- **GET /patient/dashboard** — Shows recent 5 diagnoses + stats (total, benign, malignant)
- **POST /patient/upload** — Handles image upload:
  - Validates file type (png, jpg, jpeg, tif, bmp)
  - Saves with UUID filename to prevent collisions
  - Runs `model.predict_from_path()` for AI analysis
  - Saves Diagnosis record to database
  - Redirects to result page
- **GET /patient/result/<id>** — Shows prediction results:
  - Displays: prediction, confidence, probability bars, risk level
  - If malignant: shows available doctors for consultation
- **POST /patient/request-consultation/<diag_id>/<doc_id>** — Creates consultation request
- **GET /patient/history** — Paginated diagnosis history (10 per page)

### 5.10 app/routes/doctor.py (Doctor Features)
- **GET /doctor/dashboard** — Shows pending + active consultations with stats
- **POST /doctor/consultation/<id>/accept** — Accepts a consultation, sets started_at
- **POST /doctor/consultation/<id>/complete** — Completes consultation:
  - Saves doctor notes to both Consultation and Diagnosis records
  - Sets reviewed_by_id on diagnosis
- **GET /doctor/patients** — Lists completed consultation history
- **POST /doctor/toggle-availability** — Toggles is_available flag
- **GET /doctor/diagnosis/<id>** — Views a specific diagnosis in detail

### 5.11 app/routes/chat.py (Real-Time Chat)
- **GET /chat/room/<id>** — Renders chat room, marks messages as read
- **SocketIO 'join'** — Joins a room and broadcasts status
- **SocketIO 'leave'** — Leaves a room and broadcasts status
- **SocketIO 'send_message'** — Saves message to DB, auto-updates consultation to "in_progress", broadcasts to room

### 5.12 train.py (Model Training Script)
Full training pipeline (run separately from the web app):
1. Collects image paths from BreakHis dataset structure
2. Loads and preprocesses images (resize to 224×224, normalize to [0,1])
3. Splits data: 70% train / 15% validation / 15% test (stratified)
4. Phase 1: Pre-trains CNN with frozen EfficientNetB0 (30 epochs)
5. Phase 2: Fine-tunes top 30 EfficientNetB0 layers (15 epochs)
6. Phase 3: Extracts features and trains SVM
7. Saves models, metrics, training curves, and confusion matrix

### 5.13 create_demo_model.py
Creates lightweight placeholder models for demo purposes:
- Builds the same CNN architecture but doesn't train on real data
- Trains SVM on random synthetic data
- Saves all 3 files (CNN, SVM, Scaler) so the web app can start

---

## 6. DATABASE SCHEMA

```
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│     users        │     │  patient_profiles  │     │   doctor_profiles    │
├──────────────────┤     ├────────────────────┤     ├──────────────────────┤
│ id (PK)          │────▶│ id (PK)            │     │ id (PK)              │
│ email            │     │ user_id (FK)       │     │ user_id (FK)         │
│ username         │     │ dob                │     │ specialization       │
│ password_hash    │     │ gender             │     │ hospital             │
│ role             │     │ address            │     │ license_number       │
│ full_name        │     │ medical_history    │     │ years_experience     │
│ phone            │     └────────┬───────────┘     │ bio                  │
│ created_at       │              │                  │ is_available         │
│ is_active        │              │                  └──────────┬───────────┘
│ avatar           │              │                             │
└──────────────────┘              │                             │
                                  ▼                             ▼
                    ┌─────────────────────┐      ┌─────────────────────────┐
                    │     diagnoses       │      │     consultations       │
                    ├─────────────────────┤      ├─────────────────────────┤
                    │ id (PK)             │─────▶│ id (PK)                 │
                    │ patient_id (FK)     │      │ diagnosis_id (FK)       │
                    │ image_filename      │      │ patient_id (FK)         │
                    │ prediction          │      │ doctor_id (FK)          │
                    │ is_malignant        │      │ status                  │
                    │ confidence          │      │ scheduled_at            │
                    │ prob_benign         │      │ started_at              │
                    │ prob_malignant      │      │ ended_at                │
                    │ risk_level          │      │ notes                   │
                    │ doctor_notes        │      │ created_at              │
                    │ reviewed_by_id (FK) │      └────────────┬────────────┘
                    │ created_at          │                   │
                    └─────────────────────┘                   ▼
                                                ┌─────────────────────────┐
                                                │    chat_messages        │
                                                ├─────────────────────────┤
                                                │ id (PK)                 │
                                                │ consultation_id (FK)    │
                                                │ sender_id (FK)          │
                                                │ content                 │
                                                │ timestamp               │
                                                │ is_read                 │
                                                └─────────────────────────┘
```

---

## 7. APPLICATION WORKFLOW

### 7.1 Patient Flow
```
Register as Patient → Login → Upload Histopathology Image
    → AI Analyzes Image (CNN extracts features → SVM classifies)
    → View Result (Benign/Malignant + Confidence + Risk Level)
    → If Malignant: Request Consultation with Available Doctor
    → Chat with Doctor in Real-Time
    → View Diagnosis History
```

### 7.2 Doctor Flow
```
Register as Doctor → Login → View Dashboard
    → See Pending Consultation Requests
    → Review AI Diagnosis (image + probabilities)
    → Accept Consultation → Chat with Patient
    → Complete Consultation (add clinical notes)
    → Toggle Availability On/Off
```

### 7.3 AI Inference Flow (What Happens When Patient Uploads Image)
```
1. Image file saved to app/static/uploads/ with UUID filename
2. Image loaded via OpenCV, converted BGR → RGB
3. Resized to 224×224 pixels
4. Passed through CNN feature extractor → 256-D vector
5. Features scaled by StandardScaler
6. SVM predicts class + probability
7. Probability mapped to risk level
8. Results + recommendations saved to database
9. Patient sees result page with all details
```

---

## 8. KEY DESIGN PATTERNS

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **App Factory** | `app/__init__.py` | Creates Flask app with configurable settings |
| **Blueprint** | All route files | Modular route organization |
| **Singleton** | `get_model()` | Load ML model once, reuse across requests |
| **Repository/ORM** | SQLAlchemy models | Database abstraction layer |
| **Transfer Learning** | EfficientNetB0 | Leverage ImageNet pre-trained weights |
| **Hybrid ML** | CNN + SVM | Combine deep features with classical classifier |
| **Decorator** | `_require_patient`, `_require_doctor` | Role-based access control |
| **Observer** | SocketIO events | Real-time message broadcasting |

---

## 9. SECURITY FEATURES

- Password hashing using Werkzeug's PBKDF2 algorithm
- CSRF protection via Flask-WTF
- Role-based access control (patient vs. doctor routes)
- Secure file naming with UUID to prevent path traversal
- File type validation (only image formats allowed)
- Max upload size limit (16 MB)
- Session management with configurable lifetime (12 hours)

---

## 10. HOW TO RUN

### Step 1: Setup Virtual Environment
```bash
cd C:\breast_cancer_detection
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create Demo Model (if no trained model exists)
```bash
python create_demo_model.py
```

### Step 4: Run the Application
```bash
python run.py
```

### Step 5: Access the Application
Open browser → http://localhost:5000

### Step 6: Train Real Model (Optional — requires BreakHis dataset)
```bash
python train.py --dataset /path/to/BreaKHis_v1 --magnification ALL --epochs 30
```

---

## 11. SUMMARY

BreastGuard AI combines modern deep learning (EfficientNetB0 CNN) with classical machine
learning (SVM) to create a powerful breast cancer detection tool. The Flask web application
provides an intuitive interface for patients to upload histopathology images and receive
instant AI-powered analysis, while enabling doctors to review results and provide real-time
consultations via WebSocket chat. The hybrid CNN+SVM approach is designed to achieve ≥90%
accuracy on the BreakHis dataset, making it a valuable decision-support tool for early
cancer detection.

---

*Note: This tool is for educational and research purposes only. It does not replace
professional medical diagnosis. Always consult a qualified physician.*
