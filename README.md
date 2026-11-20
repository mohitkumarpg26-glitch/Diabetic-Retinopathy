# Evaluating Nominal and Ordinal Models for Diabetic Retinopathy Severity Grading

### A Data Audit and Validation-Selected Decoding Study

**Authors:** Mohit Kumar and Malhaar Sinha  
**Department:** Computer Science and Engineering  
**Institution:** Netaji Subhas University of Technology (NSUT), New Delhi, India

**Submitted To:** Dr. Ankush Jain  
**Email:** ankush.jain@nsut.ac.in

## 1. Project Overview

This research investigates five-grade diabetic retinopathy (DR) severity classification using deep learning.

We compare two independently trained models using a common DenseNet121 backbone:

1. A class-weighted nominal classifier using five output logits.
2. A cumulative ordinal regression model using ordered thresholds.

The study focuses on data integrity, model calibration, classification accuracy, ordinal error severity, and ethical considerations in medical AI.

Rather than proposing a new architecture, the research evaluates how model formulation and validation-selected decision rules affect diabetic retinopathy grading.

## 2. Research Objectives

- Audit the APTOS 2019 dataset for exact duplicates and conflicting labels.
- Compare nominal and ordinal DenseNet121 classification approaches.
- Evaluate the effect of validation-selected decoding rules.
- Analyze severe undergrading and ordinal classification errors.
- Examine the limitations and ethical implications of automated medical image classification.
- Provide an interactive Streamlit demonstration of model predictions.

## 3. Dataset

**Dataset:** APTOS 2019 Blindness Detection

**Source:** https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

The original dataset contains 3,662 labeled retinal fundus images across five severity grades.

| Grade | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

### Data Integrity Audit

SHA-256 hashing identified exact duplicates and conflicting-label groups.

- 96 redundant copies were removed.
- 62 images belonging to conflicting-label duplicate groups were excluded.
- 3,504 images remained after cleaning.

### Dataset Split

| Partition | Images |
|---|---:|
| Training | 2,450 |
| Validation | 523 |
| Testing | 531 |
| **Total** | **3,504** |

The dataset was divided using stratified 70/15/15 splitting with random seed 42.

The audit addressed byte-identical duplicates but could not establish the absence of near-duplicates or patient-level overlap.

## 4. Model Architecture

Both model variants use an ImageNet-pretrained DenseNet121 backbone.

### Model A: Nominal Classification

- DenseNet121 feature extractor
- Five-class classification head
- Class-weighted categorical cross-entropy
- Softmax prediction probabilities
- Raw argmax and validation-selected class-logit adjustment

### Model B: Ordinal Classification

- DenseNet121 feature extractor
- Scalar severity score
- Four ordered thresholds
- Class-weighted cumulative binary cross-entropy
- Raw argmax and validation-selected expected-grade decoding

The nominal and ordinal models were trained separately. No CBAM attention module was added.

## 5. Image Preprocessing

The preprocessing pipeline consists of:

1. RGB conversion
2. Aspect-ratio-preserving resizing
3. Black padding to 224 × 224 pixels
4. ImageNet mean and standard-deviation normalization
5. Random horizontal flipping during training

No additional lesion-specific enhancement or Gaussian filtering was applied.

## 6. Training Configuration

| Parameter | Configuration |
|---|---|
| Backbone | DenseNet121 |
| Initialization | ImageNet pretrained |
| Learning rate | 0.0001 |
| Batch size | 4 |
| Optimizer | Adam |
| Training epochs | 10 |
| Checkpoint selection | Validation macro F1 |
| Baseline selected epoch | 4 |
| Ordinal selected epoch | 2 |
| Dataset split seed | 42 |

### Research Environment

- Python 3.12
- PyTorch 2.4.1
- torchvision 0.19.1
- torch-directml 0.2.5
- NumPy 1.26.4
- scikit-learn 1.7.2
- Streamlit 1.67.0
- Windows 11
- AMD Radeon RX 6600 (8 GB)

Some DirectML operations used CPU fallback.

## 7. Experimental Results

The following results were reported on the fixed 531-image APTOS test partition.

| Model | Accuracy | Macro F1 | QWK |
|---|---:|---:|---:|
| Nominal baseline (raw) | 80.23% | 0.6147 | 0.8585 |
| Nominal baseline (adjusted) | **82.11%** | **0.6510** | 0.8821 |
| Ordinal model (raw) | 56.87% | 0.2405 | 0.6271 |
| Ordinal model (tuned) | 78.15% | 0.5669 | **0.8844** |

### Key Findings

**Highest classification accuracy:** The validation-adjusted nominal DenseNet121 classifier achieved 82.11% accuracy.

**Highest quadratic weighted kappa:** The ordinal model with validation-selected cutoffs achieved 0.8844 QWK.

**Severe undergrading among true grades 2–4:**

- Raw nominal baseline: 9.57%
- Adjusted nominal baseline: 7.66%
- Raw ordinal model: 49.76%
- Tuned ordinal model: 6.70%

The tuned ordinal model reduced severe undergrading but had low grade-4 recall (11.9%). Therefore, stronger aggregate QWK does not imply consistently better classification across every severity grade.

These findings are exploratory rather than independently confirmed improvements.

## 8. Validation-Selected Decision Rules

Two post-training decision strategies were investigated.

### Nominal Class-Logit Adjustment

Offsets for grades 1–4:

`[0.2193, 1.1021, 1.4399, 0.8963]`

These offsets were selected using validation predictions and applied without retraining the baseline model.

### Ordinal Threshold Calibration

Expected-grade decoding cutoffs:

`[0.6095, 1.1196, 2.1855, 3.6469]`

The cutoffs were selected using validation QWK optimization.

The adjustments are components of the predictive pipeline and should be disclosed when reporting results.

## 9. Project Files

```text
Diabetic-Retinopathy/
│
├── README.md
├── requirements.txt
├── START_HERE.md
├── VALIDATION.md
│
├── app.py
├── retina_project.py
├── calibrate_accuracy_offsets.py
├── calibrate_ordinal_thresholds.py
├── decision_adjustment.py
├── evaluate_flip_tta.py
├── download_aptos.py
├── generate_pdf.py
│
├── src/
├── results/
│
├── main.pdf
└── supporting literature.pdf
```

This structure represents the project files shown in the GitHub repository. The full research paper can be added as a separate PDF.

## 10. Installation and Execution

This repository provides the source code, research documentation, and Streamlit application for diabetic retinopathy severity grading.

The trained DenseNet121 baseline checkpoint is distributed separately through GitHub Releases because of its file size.

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohitkumarpg26-glitch/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy
```

### Step 2: Create a Python Virtual Environment

Python 3.12 was used during development.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The locally tested research environment used:

- Python 3.12
- PyTorch 2.4.1
- torchvision 0.19.1
- Streamlit 1.67.0

Other project dependencies are documented in `requirements.txt`.

### Step 4: Download the Trained Model Checkpoint

**GitHub Release:**  
https://github.com/mohitkumarpg26-glitch/Diabetic-Retinopathy/releases/tag/v1.0.0

Download the release asset:

**File:** `best.pt`  
**Size:** Approximately 80.4 MB  
**Model:** Fine-tuned DenseNet121 nominal baseline

This checkpoint was used for the locally verified Streamlit inference demonstration.

### Step 5: Create the Model Directory

After cloning the repository, create the following directories:

```text
Diabetic-Retinopathy/
└── runs/
    └── baseline_finetune/
```

Place the downloaded `best.pt` inside `baseline_finetune`.

The expected project structure is:

```text
Diabetic-Retinopathy/
│
├── README.md
├── requirements.txt
├── START_HERE.md
├── VALIDATION.md
├── app.py
├── retina_project.py
├── src/
├── results/
│
└── runs/
    └── baseline_finetune/
        └── best.pt
```

**Important:** The `runs` directory is not included in the GitHub source repository. Users must create the required folders and download the checkpoint separately.

### Step 6: Launch the Streamlit Application

From the repository's root directory, execute:

```bash
python -m streamlit run app.py
```

The application should become accessible at:

http://localhost:8501

### Step 7: Test Model Inference

1. Open the Streamlit application.
2. Navigate to the **Explore an image** tab.
3. Upload a de-identified retinal fundus image in PNG or JPEG format.
4. Click **Estimate research grade**.
5. Review the predicted severity grade and model score visualization.

The application supports five severity grades:

| Grade | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Prediction scores are research model outputs and should not be interpreted as clinically calibrated diagnostic confidence.

### Local Execution Verification

The application was tested locally on Windows with a configured Python environment and the trained baseline checkpoint.

The local test confirmed:

- Streamlit application startup
- Retinal image upload
- Image preprocessing and preview
- PyTorch model inference
- Adjusted baseline grade prediction
- Prediction score visualization

**Example local inference result:**

- Original baseline prediction: Grade 2 — Moderate
- Adjusted baseline prediction: Grade 3 — Severe
- Checkpoint: `baseline_finetune`, selected at epoch 4

This is an example of application functionality, not an independently validated clinical prediction.

### Reproducibility Status

The application successfully executed locally with the required dependencies and checkpoint.

Independent execution from a fresh GitHub clone has not yet been verified.

The publicly available release includes the baseline checkpoint used for the demonstration. Additional training datasets and other experimental model checkpoints are not included in this release.

For research methodology and validation details, refer to `START_HERE.md`, `VALIDATION.md`, and the accompanying research paper.

### Important Disclaimer

This implementation is intended for academic research and educational demonstration only.

The system is not clinically validated and must not be used independently for medical diagnosis, referral, or treatment decisions.

## 11. Evaluation Metrics

The study reports:

- Accuracy
- Macro F1-score
- Quadratic weighted kappa (QWK)
- Mean absolute grade error (MAE)
- Severe undergrading
- Per-grade precision and recall
- Confusion matrices

Severe undergrading is defined as predicting at least two grades below the reference grade.

The paper separately reports this measure across all images and among images with true grades 2–4.

## 12. Comparison with Previous Research

The research compares its findings descriptively with the published work of Farag et al., which reported:

- Five-class accuracy: 82.0%
- QWK: 0.888
- Architecture: DenseNet169 with CBAM

Our adjusted DenseNet121 nominal model achieved:

- Accuracy: 82.11%
- QWK: 0.8821

Because the studies differ in architecture, preprocessing, dataset partitioning, and evaluation protocol, these results do not establish overall superiority.

## 13. Ethical Considerations

### Clinical Safety

The models have not undergone prospective clinical validation. Their outputs must not be used independently for diagnosis, referral, or treatment.

### Data Integrity

Exact duplicate and conflicting-label auditing improves data transparency, but near-duplicate and patient-level overlap remain unresolved.

### Fairness

The dataset does not provide the patient-level demographic attributes needed for a demographic fairness analysis.

### Model Transparency

Prediction adjustments and calibration parameters must be disclosed to make the reported outputs interpretable.

### Risk of Undergrading

Undergrading may be important in medical image analysis, but the study does not measure actual clinical harm or patient outcomes.

## 14. Research Limitations

- Experiments use one dataset and one fixed random split.
- The test set had been inspected during project development.
- Minority-grade test populations are relatively small.
- Near-duplicate and patient-level overlap could not be assessed.
- Validation-based model selection and decoding introduce selection dependence.
- No external or prospective clinical validation was conducted.
- Reported probabilities have not been established as clinically calibrated.

Accordingly, reported comparisons are exploratory image-level results, not evidence of clinical safety or generalization.

## 15. Future Work

- External validation using independent retinal image datasets
- Patient-level splitting and overlap assessment
- Repeated or nested validation
- Improved minority-grade evaluation
- Controlled preprocessing experiments
- Cost-sensitive classification objectives
- Probability calibration assessment
- Clinician review and prospective research evaluation

## 16. Research Paper

**Title:** Evaluating Nominal and Ordinal Models for Diabetic Retinopathy Severity Grading: A Data Audit and Validation-Selected Decoding Study

**Authors:**
- Mohit Kumar
- Malhaar Sinha

**Institution:** Netaji Subhas University of Technology (NSUT), Delhi, India

The complete research paper documents the methodology, model architecture, experimental findings, confusion matrices, ethical considerations, and supporting references.

## Research Contributions and Model Comparison

This project investigates how nominal classification and ordinal regression affect five-grade diabetic retinopathy severity prediction.

### Key Research Contributions

- *Data Integrity:* Audited the APTOS 2019 dataset for byte-identical duplicates and conflicting labels, retaining 3,504 images.
- *Model Comparison:* Evaluated independently trained nominal and ordinal DenseNet121 classifiers using fixed data partitions.
- *Decision Calibration:* Investigated validation-selected logit adjustments and ordinal thresholds without retraining model weights.
- *Error Analysis:* Compared accuracy, macro F1, quadratic weighted kappa, and severe undergrading across model configurations.
- *Responsible AI:* Documented dataset limitations, prediction uncertainty, and the need for independent clinical validation.

### Research Findings

The adjusted nominal model achieved the highest classification accuracy (82.11%), while the ordinal model with validation-selected thresholds achieved the highest quadratic weighted kappa (0.8844).

These exploratory findings demonstrate that the choice of decoding strategy can substantially influence evaluation metrics and classification errors.

The results do not establish clinical effectiveness or superiority over previously published approaches.

## 17. Academic Submission

**Submitted To:**  
Dr. Ankush Jain  
Netaji Subhas University of Technology (NSUT)  
Email: ankush.jain@nsut.ac.in

**Submitted By:**
- Mohit Kumar
- Malhaar Sinha

**GitHub Repository:**  
https://github.com/mohitkumarp26-glitch/Diabetic-Retinopathy

## 18. Disclaimer

This project is intended exclusively for academic research, experimentation, and educational demonstration.

The models have not received clinical validation, regulatory approval, or professional medical assessment. Predictions must not be used to make clinical decisions.

---

**Research Project — Department of Computer Science and Engineering, NSUT**
