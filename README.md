# AI-Powered Prior Authorization Automation System

This project is a proof-of-concept demonstrating how Large Language Models (LLMs) can be used to automate the complex and time-consuming process of healthcare prior authorization (PA). The system intelligently analyzes patient electronic health records (EHR), checks them against payer-specific rules, and automates the submission process.

## ✨ Core Features

*   **Automated PA Requirement Check**: Instantly determines if a prescribed drug requires prior authorization for a specific payer.
*   **AI-Powered Gap Analysis**: Leverages a Natural Language Inference (NLI) model to understand unstructured clinical notes and identify if criteria like "failed therapy" have been met.
*   **Structured Data Validation**: Checks for required diagnoses (ICD-10 codes) and validates lab results against payer-defined value ranges (e.g., `min_value`, `max_value`).
*   **Dynamic Statement Generation**: Uses a text generation model to write a professional and coherent "Statement of Medical Necessity" (SMN) by synthesizing the patient's clinical data.
*   **Simulated End-to-End Workflow**: Models the entire PA lifecycle from initial check, to gap analysis, form population, submission (via mock API, Portal, or eFax), and status tracking.

## 🧠 The AI Core: Leveraging Hugging Face Transformers

This system utilizes two distinct types of LLMs to handle different aspects of the PA process, showcasing a powerful multi-agent approach.

### 1. Clinical Note Analysis (Zero-Shot Classification)
To understand the nuances of unstructured text in a doctor's notes, we can't rely on simple keyword searching. For example, determining if a patient "failed" a previous medication requires contextual understanding.

*   **Model**: `facebook/bart-large-mnli`
*   **Task**: Natural Language Inference (NLI) / Zero-Shot Classification.
*   **Function**: The `llm_check_failed_therapy` function uses this model to test a hypothesis. It presents the clinical note as a "premise" and asks the model to classify whether a "hypothesis" (e.g., *"The patient has failed treatment on Metformin"*) is true based on that premise. This allows the system to make an informed decision without being explicitly trained on thousands of clinical notes.

### 2. Statement of Medical Necessity Generation (Text-to-Text Generation)
Once all criteria are verified, a justification needs to be written for the payer. This needs to be more than just a list of facts; it should be a professional, human-readable paragraph.

*   **Model**: `google/flan-t5-large`
*   **Task**: Text-to-Text Generation.
*   **Function**: The `llm_generate_smn` function takes the structured, verified clinical evidence (e.g., "Diagnosis E11.9 met," "Lab HbA1c 8.0 met criteria," "Failed therapy on Metformin confirmed") and uses this powerful generative model to synthesize it into a formal Statement of Medical Necessity.

## ⚙️ Workflow Overview

The script follows a logical, step-by-step process for each PA request:

1.  **[Check] Is PA Required?**
    *   Looks up the Payer/Drug combination in the `PAYER_RULES_DB`.
    *   If `False`, the process stops. ✅

2.  **[Analyze] AI Gap Analysis**
    *   If PA is required, the system compares the patient's EHR against the payer's criteria.
    *   It checks structured data (diagnoses, labs).
    *   It uses the NLI model (`bart-large-mnli`) to interpret unstructured clinical notes.
    *   **If Gaps are Found**: The process HALTS and reports exactly what criteria are met and what is missing, providing clear guidance for the provider. ❌
    *   **If No Gaps are Found**: All criteria are met. The process continues. ✅

3.  **[Submit] Populate & Submit Form**
    *   If the rules require a Statement of Medical Necessity, the system uses the generative model (`flan-t5-large`) to write one. 🤖
    *   It populates a final submission form with all patient data and the generated statement.
    *   It "submits" the form using the payer's preferred method (simulated API call, Portal post, or eFax).

4.  **[Track] Monitor Submission Status**
    *   The system simulates tracking the submission until a final status (`Approved` or `Denied`) is received.

## 🚀 Getting Started

### Prerequisites

*   Python 3.11.9
*   `pip` for installing packages

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/czhang87/prior_authorization.git
    cd prior_authorization
    ```

2.  **Install the required Python libraries:**
    This project depends on `transformers` from Hugging Face and a backend ML framework like `torch` (PyTorch) or `tensorflow`. The script is written for PyTorch.

    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** The first time you run the script, `transformers` will download the LLM models (several GBs). This is a one-time process, and subsequent runs will be much faster.

### Running the Demo

To run the simulation with the predefined patient scenarios, simply execute the Python script:

```bash
python prior_authorization.py
```

## 📋 Example Output

Here are two examples of what the script's output looks like.

### Scenario 1: Successful Submission

The patient meets all criteria, and the submission is automated.

```
============================================================
STARTING PRIOR AUTH FLOW FOR: Susan Jones | DRUG: Ozemra
============================================================

[Step 1] RESULT: Prior Authorization IS required for Ozemra.

[Step 2] AI is performing Gap Analysis on patient's EHR...

   (Classifier LLM) Testing hypotheses for 'Jardiance' against the clinical note...
   (Classifier LLM Insight) Response: {'failed': True, 'reason': "Top hypothesis: 'This patient has experienced treatment failure or adverse effects from Jardiance.' with confidence 0.89."}

[Step 2] RESULT: NO GAPS FOUND. All criteria met.
  - [MET] Diagnosis criteria met: E11.9
  - [MET] Failed therapy criteria met: Jardiance

[Step 3] Proceeding to automated submission...

   (Generator LLM - FLAN-T5) Writing Statement of Medical Necessity for Ozemra...

   (SMN Generated): "Susan Jones has been diagnosed with E11.9 and failed therapy with Jardiance. Ozemra is a new medication that can be used to treat Susan Jones."

[Submission] Populated form for Susan Jones:
{
  "patient_id": "PID-006",
  "drug_name": "Ozemra",
  "payer_id": "Payer_B",
  "clinical_justification": {
    "diagnoses": [
      "E11.9"
    ],
    "labs": [
      {
        "name": "HbA1c",
        "value": 7.6
      }
    ],
    "notes": "Patient previously failed on Jardiance due to side effects. She was started on Metformin 3 months ago and is tolerating it well."
  },
  "statement_of_medical_necessity": "Susan Jones has been diagnosed with E11.9 and failed therapy with Jardiance. Ozemra is a new medication that can be used to treat Susan Jones."
}
[Submission] Submitting via preferred method: PORTAL
[Submission] Success! Tracking ID: PA-38570

[Step 4] Handing off to automated status tracker...

[Status Check] Tracking submission PA-38570...
--> Current status: Approved
--> Final status: APPROVED

PROCESS COMPLETE
```

### Scenario 2: Gaps Found (Submission Halted)

The patient's lab values do not meet the payer's criteria.

```
============================================================
STARTING PRIOR AUTH FLOW FOR: Eva Garcia | DRUG: GlycoLow
============================================================

[Step 1] RESULT: Prior Authorization IS required for GlycoLow.

[Step 2] AI is performing Gap Analysis on patient's EHR...

[Step 2] RESULT: GAPS FOUND! Submission halted.
Provider Guidance:
  - [MISSING] Lab result of Glucose 80 misses criteria (<= 70).

ACTION: Please address missing items before resubmitting.
PROCESS HALTED
```

## 📂 Code Structure

*   `prior_authorization.py`: The main executable script.
*   **Mock Databases**:
    *   `PAYER_RULES_DB`: A dictionary simulating a database of payer rules for different drugs.
    *   `PAYER_SUBMISSION_PROFILES`: Defines the preferred contact method for each payer.
    *   `PATIENT_EHR_*`: Dictionaries representing simplified patient records.
*   **LLM Functions**:
    *   `llm_check_failed_therapy()`: Interfaces with the NLI model.
    *   `llm_generate_smn()`: Interfaces with the text generation model.
*   **`PriorAuthAISystem` Class**: The core class that orchestrates the entire workflow.

## ⚖️ Disclaimer

This is a conceptual demonstration and a proof-of-concept. It uses mock data and is **NOT intended for use in a real clinical or production environment**. The primary purpose is to showcase the potential of LLMs in automating complex administrative tasks in healthcare.