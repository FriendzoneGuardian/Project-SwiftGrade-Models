# 🧠 SwiftGrade Models Branch

Welcome to the `models` branch of **SwiftGrade** — the machine-learning backbone that powers our OCR + NLP hybrid grading engine. If you're here, you're either trying to debug a .h5 crisis, contribute a new model, or wonder how an essay rejection parser ended up being this... powerful.

---

## 📦 Contents of this Branch

This branch contains:

- 🔤 **OCR Modules** – Optical Character Recognition pre-processors, based on Tesseract / EasyOCR / Custom pipelines.
- 🧾 **NLP Modules** – Essay analyzers, rejection reason detectors, and possibly an emotional damage classifier.  
- 🧠 **Trained ML Models**:
  - `.h5` and `.tflite` variants for backend and mobile deployment
  - Note: `.h5` is *NOT* pushed to GitHub (see below)
- ⚙️ **Preprocessing Scripts** – Utilities for text cleaning, tokenization, stopword handling, and syntactic armageddon.

---

## 🤖 Model Deployment Philosophy

> “Don’t bring a 112MB `.h5` to a GitHub commit.”

We learned this the hard way. TensorFlow is powerful, but GitHub has a 100MB file limit. That means:
- `.h5` and other model binaries are **excluded via `.gitignore`**
- Use `.tflite` for lightweight mobile deployment (in `dev` branch integration)

For full `.h5` or larger files, refer to:
- 🔗 Internal Google Drive / HuggingFace Model Repo
- 📁 `ModelBackEnd/SwiftGrade_Datasets/` (for local dev only, never push upstream)

---

## 📊 Contribution Guidelines
- Work on feature branches like models-ocr-improvements, models-nlp-cleanup

- Commit early, commit often — but test before pushing

- Run *black* or *autopep8* if your code starts looking like spaghetti

- Humor is allowed in comments. Production bugs are not.

---

## 🔥 Warning Log
*“That moment you pushed a 112MB .h5 file and GitHub said ‘no.’”*

If you:

1. See 10,000+ changes in git status → you probably added venv by mistake.

2. Can’t push a model → it’s too thicc, use .gitignore and external storage.

---

## 🧑‍🔬 Acknowledgments
Shoutout to:

Teammates in dev – for turning our ugly matrix math into beautiful UI

Advisers – for tolerating this Frankenstein of NLP and OCR

## 📜 License
Academic research use only. Not for commercial or heartbreak therapy resale.
If you improve the accuracy, please submit a PR ~~(and a meme)~~.
