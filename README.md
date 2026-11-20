# Evaluating Nominal and Ordinal Models for Diabetic Retinopathy Severity Grading

### A Data Audit and Validation-Selected Decoding Study

**Authors:** Mohit Kumar and Malhaar Sinha  
**Department:** Computer Science and Engineering  
**Institution:** Netaji Subhas University of Technology (NSUT), Delhi, India

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

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohitkumarp26-glitch/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application is designed to support retinal image upload, preprocessing, checkpoint-based inference, and grade-probability visualization.

**Important:** The dataset and trained model weights are not included in the public repository. Required checkpoint paths and model availability must be configured before inference can run successfully.

Consult `START_HERE.md` for additional project-specific instructions.

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
