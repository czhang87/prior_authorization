import time
import random
import json

# --- IMPORTANT: Import and initialize the Hugging Face pipelines ---
try:
    from transformers import pipeline

    print("Initializing Hugging Face pipelines...")
    # Pipeline 1: For understanding and classifying notes (NLI task)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Pipeline 2 : Using a model for text generation ---
    text_generator = pipeline("text2text-generation", model="google/flan-t5-large")

    print("Pipelines initialized successfully.")
except ImportError:
    print("Error: The 'transformers' or 'torch' library is not installed.")
    print("Please run: pip install transformers torch")
    exit()

# --- MOCK DATABASES ---
PAYER_RULES_DB = {
    ("Payer_A", "Ozemra"): {
        "requires_pa": True,
        "criteria": {
            "required_diagnosis": "E11.9",
            "failed_therapy": "Metformin",
            "required_lab": {"name": "HbA1c", "min_value": 7.5},
            "requires_smn": True,
        },
    },
    ("Payer_B", "Ozemra"): {
        "requires_pa": True,
        "criteria": {
            "required_diagnosis": "E11.9",
            "failed_therapy": "Jardiance",
            "requires_smn": True,
        },
    },
    ("Payer_A", "Amoxicillin"): {"requires_pa": False},
    ("Payer_A", "GlycoLow"): {
        "requires_pa": True,
        "criteria": {
            "required_lab": {"name": "Glucose", "max_value": 70}
        },  # Rule with MAX value only
    },
    ("Payer_A", "RenalCare"): {
        "requires_pa": True,
        "criteria": {
            "required_lab": {"name": "Creatinine", "min_value": 0.6, "max_value": 1.2}
        },  # Rule with a RANGE
    },
}

PAYER_SUBMISSION_PROFILES = {
    "Payer_A": {"method": "API", "endpoint": "https://api.payera.com/priorauth"},
    "Payer_B": {"method": "PORTAL", "url": "https://portal.payerb.com/submit"},
    "Payer_C": {"method": "EFAX", "number": "1-800-555-1234"},
}

# --- SIMULATED PATIENT DATA ---
PATIENT_EHR_5 = {
    "patient_id": "PID-005",
    "name": "Chen Wei",
    "payer": "Payer_A",
    "diagnoses": ["E11.9"],
    "labs": [{"name": "HbA1c", "value": 8.0}],
    "notes": "The patient's response to Metformin has been unsatisfactory due to poor glycemic control, so we are discontinuing it.",
}
PATIENT_EHR_6 = {
    "patient_id": "PID-006",
    "name": "Susan Jones",
    "payer": "Payer_B",
    "diagnoses": ["E11.9"],
    "labs": [{"name": "HbA1c", "value": 7.6}],
    "notes": "Patient previously failed on Jardiance due to side effects. She was started on Metformin 3 months ago and is tolerating it well.",
}
PATIENT_EHR_7 = {
    "patient_id": "PID-007",
    "name": "Mary Smith",
    "payer": "Payer_A",
    "diagnoses": ["J02.9"],
    "labs": [],
    "notes": "The patient needs antibiotics to control a throat infection.",
}

PATIENT_EHR_8 = {
    "patient_id": "PID-008",
    "name": "David Miller",
    "payer": "Payer_A",
    "diagnoses": [],
    "labs": [{"name": "Glucose", "value": 65}],
    "notes": "Routine checkup.",  # Should PASS GlycoLow
}
PATIENT_EHR_9 = {
    "patient_id": "PID-009",
    "name": "Eva Garcia",
    "payer": "Payer_A",
    "diagnoses": [],
    "labs": [{"name": "Glucose", "value": 80}],
    "notes": "Routine checkup.",  # Should FAIL GlycoLow
}
PATIENT_EHR_10 = {
    "patient_id": "PID-010",
    "name": "Frank White",
    "payer": "Payer_A",
    "diagnoses": [],
    "labs": [{"name": "Creatinine", "value": 1.0}],
    "notes": "Kidney function test.",  # Should PASS RenalCare
}


# --- LLM Functions ---
def llm_check_failed_therapy(clinical_note, drug_name):
    print(
        f"\n   (Classifier LLM) Testing hypotheses for '{drug_name}' against the clinical note..."
    )
    candidate_labels = [
        f"This patient has experienced treatment failure or adverse effects from {drug_name}.",
        f"This patient is tolerating or responding well to {drug_name}.",
    ]
    result = classifier(
        clinical_note,
        candidate_labels,
        hypothesis_template="The clinical note says that {}.",
    )
    top_label, top_score = result["labels"][0], result["scores"][0]
    reason = f"Top hypothesis: '{top_label}' with confidence {top_score:.2f}."
    if "failure" in top_label and top_score > 0.80:
        return {"failed": True, "reason": reason}
    else:
        return {"failed": False, "reason": reason}


def llm_generate_smn(patient_ehr, drug_name, met_criteria):
    print(
        f"\n   (Generator LLM - FLAN-T5) Writing Statement of Medical Necessity for {drug_name}..."
    )

    system_prompt = "You are a medical administrator writing a Statement of Medical Necessity. Your task is to synthesize the provided clinical data into a professional, fluent paragraph justifying the requested medication.\n"
    formatted_criteria = "\n- ".join(met_criteria)
    user_prompt = (
        "VERIFIED CLINICAL POINTS:\n"
        f"- Patient Name: {patient_ehr['name']}\n"
        f"- {formatted_criteria}\n\n"
        f"- Proposed New Medication: {drug_name}\n\n"
        "INSTRUCTION: Write a professional Statement of Medical Necessity stating all met criteria. The statement must be a natural, well-written paragraph."
    )
    prompt = system_prompt + user_prompt

    generated = text_generator(prompt, max_new_tokens=100, repetition_penalty=1.2)
    statement = generated[0]["generated_text"].strip()
    return statement


class PriorAuthAISystem:
    def __init__(self, payer_rules, payer_profiles):
        self.payer_rules = payer_rules
        self.payer_profiles = payer_profiles

    def check_if_pa_required(self, patient_ehr, drug_name):
        payer = patient_ehr["payer"]
        rule = self.payer_rules.get((payer, drug_name), {"requires_pa": False})
        return rule["requires_pa"]

    def extract_clinical_data(self, patient_ehr, criteria):
        evidence = {}
        if (
            "required_diagnosis" in criteria
            and criteria["required_diagnosis"] in patient_ehr["diagnoses"]
        ):
            evidence["diagnosis_met"] = True

        if "required_lab" in criteria:
            lab_req = criteria["required_lab"]
            for lab in patient_ehr.get("labs", []):
                if lab.get("name") == lab_req.get("name"):
                    # Check if the lab value meets all specified conditions
                    is_met = True
                    if (
                        "min_value" in lab_req
                        and lab.get("value", 0) < lab_req["min_value"]
                    ):
                        is_met = False
                    if (
                        "max_value" in lab_req
                        and lab.get("value", 0) > lab_req["max_value"]
                    ):
                        is_met = False

                    if is_met:
                        evidence["lab_met"] = True
                        break  # Found a matching and valid lab, no need to check further

        if "failed_therapy" in criteria:
            failed_therapy_drug = criteria["failed_therapy"]
            llm_response = llm_check_failed_therapy(
                patient_ehr["notes"], failed_therapy_drug
            )
            print(f"   (Classifier LLM Insight) Response: {llm_response}")
            if llm_response.get("failed") is True:
                evidence["failed_therapy_met"] = True
        return evidence

    def perform_gap_analysis(self, patient_ehr, drug_name):
        payer = patient_ehr["payer"]
        rule = self.payer_rules.get((payer, drug_name))
        if not rule or not rule.get("criteria"):
            return {"gaps_found": False, "missing": [], "met": []}
        criteria = rule["criteria"]
        extracted_evidence = self.extract_clinical_data(patient_ehr, criteria)
        missing_criteria = []
        met_criteria = []
        if "required_diagnosis" in criteria and not extracted_evidence.get(
            "diagnosis_met"
        ):
            missing_criteria.append(
                f"Missing required diagnosis: {criteria['required_diagnosis']}"
            )
        elif "required_diagnosis" in criteria:
            met_criteria.append(
                f"Diagnosis criteria met: {criteria['required_diagnosis']}"
            )
        if "required_lab" in criteria:
            req = criteria["required_lab"]

            # Helper to build the condition string
            condition_parts = []
            if "min_value" in req:
                condition_parts.append(f">= {req['min_value']}")
            if "max_value" in req:
                condition_parts.append(f"<= {req['max_value']}")
            condition_str = " and ".join(condition_parts)

            if not extracted_evidence.get("lab_met"):
                for lab in patient_ehr.get("labs", []):
                    if lab.get("name") == req.get("name"):
                        missing_criteria.append(
                            f"Lab result of {lab['name']} {lab['value']} misses criteria ({condition_str})."
                        )
                        break
            else:
                # Find the actual lab value that met the criteria for better feedback
                for lab in patient_ehr.get("labs", []):
                    if lab.get("name") == req.get("name"):
                        met_criteria.append(
                            f"Lab result of {lab['name']} {lab['value']} meets criteria ({condition_str})."
                        )
                        break

        if "failed_therapy" in criteria and not extracted_evidence.get(
            "failed_therapy_met"
        ):
            missing_criteria.append(
                f"Missing evidence of failed therapy on {criteria['failed_therapy']}"
            )
        elif "failed_therapy" in criteria:
            met_criteria.append(
                f"Failed therapy criteria met: {criteria['failed_therapy']}"
            )
        return {
            "gaps_found": bool(missing_criteria),
            "missing": missing_criteria,
            "met": met_criteria,
            "full_evidence": extracted_evidence,
        }

    def populate_and_submit_form(self, patient_ehr, drug_name, analysis):
        payer = patient_ehr["payer"]
        profile = self.payer_profiles.get(payer)
        if not profile:
            return {"success": False, "message": f"No submission profile for {payer}"}

        statement_of_necessity = "Not Required"
        payer_rule = self.payer_rules.get((payer, drug_name), {})

        if payer_rule.get("criteria", {}).get("requires_smn"):
            failed_drug = payer_rule.get("criteria", {}).get(
                "failed_therapy", "a previous medication"
            )
            statement_of_necessity = llm_generate_smn(
                patient_ehr, drug_name, analysis["met"]
            )
            print(f'\n   (SMN Generated): "{statement_of_necessity}"')

        submission_form = {
            "patient_id": patient_ehr["patient_id"],
            "drug_name": drug_name,
            "payer_id": payer,
            "clinical_justification": {
                "diagnoses": patient_ehr["diagnoses"],
                "labs": patient_ehr["labs"],
                "notes": patient_ehr["notes"],
            },
            "statement_of_medical_necessity": statement_of_necessity,
        }

        print(
            f"\n[Submission] Populated form for {patient_ehr['name']}:\n{json.dumps(submission_form, indent=2)}"
        )
        method = profile["method"]
        print(f"[Submission] Submitting via preferred method: {method}")
        time.sleep(1)
        if method == "API":
            print(f"--> POST to {profile['endpoint']}")
        tracking_id = f"PA-{random.randint(10000, 99999)}"
        print(f"[Submission] Success! Tracking ID: {tracking_id}")
        return {"success": True, "tracking_id": tracking_id}

    def track_submission_status(self, tracking_id):
        print(f"\n[Status Check] Tracking submission {tracking_id}...")
        status = "Pending"
        attempts = 0
        while status == "Pending" and attempts < 5:
            time.sleep(1.5)
            status = random.choice(["Pending", "Approved", "Denied"])
            print(f"--> Current status: {status}")
            attempts += 1
        if status == "Pending":
            status = "Approved"
        print(f"--> Final status: {status.upper()}")
        return status


def run_prior_auth_flow(system, patient_ehr, drug_name):
    print(
        "=" * 60
        + f"\nSTARTING PRIOR AUTH FLOW FOR: {patient_ehr['name']} | DRUG: {drug_name}\n"
        + "=" * 60
    )
    if not system.check_if_pa_required(patient_ehr, drug_name):
        print(
            f"\n[Step 1] PASSED: Prior Authorization is NOT required for {drug_name}.\nPROCESS COMPLETE"
        )
        return
    print(f"\n[Step 1] RESULT: Prior Authorization IS required for {drug_name}.")
    print("\n[Step 2] AI is performing Gap Analysis on patient's EHR...")
    analysis = system.perform_gap_analysis(patient_ehr, drug_name)
    if analysis["gaps_found"]:
        print("\n[Step 2] RESULT: GAPS FOUND! Submission halted.")
        print("Provider Guidance:")
        for gap in analysis["missing"]:
            print(f"  - [MISSING] {gap}")
        for met in analysis["met"]:
            print(f"  - [MET] {met}")
        print(
            "\nACTION: Please address missing items before resubmitting.\nPROCESS HALTED"
        )
    else:
        print("\n[Step 2] RESULT: NO GAPS FOUND. All criteria met.")
        for met in analysis["met"]:
            print(f"  - [MET] {met}")
        print("\n[Step 3] Proceeding to automated submission...")
        submission_result = system.populate_and_submit_form(
            patient_ehr, drug_name, analysis
        )
        if submission_result["success"]:
            print("\n[Step 4] Handing off to automated status tracker...")
            system.track_submission_status(submission_result["tracking_id"])
            print("\nPROCESS COMPLETE")
        else:
            print(f"\n[Step 3] FAILED: {submission_result['message']}\nPROCESS HALTED")


if __name__ == "__main__":
    ai_system = PriorAuthAISystem(PAYER_RULES_DB, PAYER_SUBMISSION_PROFILES)
    run_prior_auth_flow(ai_system, PATIENT_EHR_5, "Ozemra")
    run_prior_auth_flow(ai_system, PATIENT_EHR_6, "Ozemra")
    run_prior_auth_flow(ai_system, PATIENT_EHR_7, "Amoxicillin")

    run_prior_auth_flow(ai_system, PATIENT_EHR_8, "GlycoLow")
    run_prior_auth_flow(ai_system, PATIENT_EHR_9, "GlycoLow")
    run_prior_auth_flow(ai_system, PATIENT_EHR_10, "RenalCare")
