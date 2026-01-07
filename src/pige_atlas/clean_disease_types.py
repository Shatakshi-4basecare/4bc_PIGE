"""Helper functions to clean and normalize disease type names."""


def clean_disease_type(disease_type: str) -> str:
    """Clean and shorten disease type names for better display."""
    if not disease_type:
        return disease_type

    if " with " in disease_type:
        disease_type = disease_type.split(" with ")[0].strip()

    if ";" in disease_type:
        disease_type = disease_type.split(";")[0].strip()

    mappings = {
        "B-Lymphoblastic Leukemia/Lymphoma": "B-Lymphoblastic Leukemia/Lymphoma",
        "T-Lymphoblastic Leukemia/Lymphoma": "T-Lymphoblastic Leukemia/Lymphoma",
        "Mature B-Cell Neoplasms": "Mature B-Cell Neoplasms",
        "Mature T and NK Neoplasms": "Mature T/NK Neoplasms",
        "Invasive Breast Carcinoma": "Breast Carcinoma",
        "Colorectal Adenocarcinoma": "Colorectal Cancer",
        "Non-Small Cell Lung Cancer": "NSCLC",
        "Small Cell Lung Cancer": "SCLC",
        "Esophagogastric Adenocarcinoma": "Esophagogastric Cancer",
        "Head and Neck Squamous Cell Carcinoma": "Head and Neck Cancer",
        "Bladder Urothelial Carcinoma": "Bladder Cancer",
        "Renal Cell Carcinoma": "Kidney Cancer",
        "Hepatocellular Carcinoma": "Liver Cancer",
        "Pancreatic Adenocarcinoma": "Pancreatic Cancer",
        "Diffuse Glioma": "Glioma",
        "Glioblastoma": "Glioblastoma",
        "Ovarian Epithelial Tumor": "Ovarian Cancer",
        "Endometrial Carcinoma": "Endometrial Cancer",
        "Cervical Squamous Cell Carcinoma": "Cervical Cancer",
        "Prostate Adenocarcinoma": "Prostate Cancer",
        "Acute Myeloid Leukemia": "AML",
        "Chronic Myeloid Leukemia": "CML",
        "Acute Lymphoblastic Leukemia": "ALL",
        "Chronic Lymphocytic Leukemia": "CLL",
        "Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma": "Undifferentiated Sarcoma",
    }

    if disease_type in mappings:
        return mappings[disease_type]

    for key, value in mappings.items():
        if disease_type.startswith(key):
            return value

    return disease_type
