import multiprocessing
import os
import pandas as pd
import tiktoken

from filelock import FileLock
from functools import partial
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Mapping

from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Input,
    Scenario,
    Instance,
    CORRECT_TAG,
    Reference,
    Output,
)

##################################
# Config
##################################
CONFIG: Dict[str, Any] = {
    "seed": 1,
    "n_shots": 0,
    "ehr_converter": "codes_only",
    "max_labels_per_task": 10_000,
    "guo": {
        "is_include_persona": True,
        "is_include_clinical_def": False,
        "is_include_code_def": False,
        "is_use_short_clinical_def": False,
        "is_include_cot": False,
    },
    "lab": {
        "is_include_persona": True,
        "is_include_clinical_def": False,
        "is_include_code_def": False,
        "is_use_short_clinical_def": False,
        "is_include_cot": False,
    },
    "new": {
        "is_include_persona": True,
        "is_include_clinical_def": True,
        "is_include_code_def": True,
        "is_use_short_clinical_def": True,
        "is_include_cot": False,
    },
}

##################################
# Task names
##################################

TASK_FULL_NAMES = {
    "new_acutemi": "Acute Myocardial Infarction",
    "new_celiac": "Celiac Disease",
    "new_hyperlipidemia": "Hyperlipidemia",
    "new_hypertension": "Hypertension",
    "new_lupus": "Systemic Lupus Erythematosus",
    "new_pancan": "Pancreatic Cancer",
    "lab_anemia": "Anemia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "guo_los": "Length of Stay",
    "guo_readmission": "30-Day Readmission",
    "guo_icu": "ICU Transfer",
}

TASK_QUESTIONS = {
    "new_acutemi": "Is this patient likely to receive a first-time diagnosis of \
    Acute Myocardial Infarction within the next year?",
    "new_celiac": "Is this patient likely to receive a first-time diagnosis of \
    Celiac Disease within the next year?",
    "new_hyperlipidemia": "Is this patient likely to receive a first-time diagnosis \
    of Hyperlipidemia within the next year?",
    "new_hypertension": "Is this patient likely to receive a first-time diagnosis \
    of Hypertension within the next year?",
    "new_lupus": "Is this patient likely to receive a first-time diagnosis of \
    Systemic Lupus Erythematosus within the next year?",
    "new_pancan": "Is this patient likely to receive a first-time diagnosis of \
    Pancreatic Cancer within the next year?",
    "lab_anemia": "If a lab test for Anemia is ordered for this patient right now, \
    will it come back back abnormal? (i.e. <120 g/L)",
    "lab_hyperkalemia": "If a lab test for Hyperkalemia is ordered for this patient right now, \
    will it come back back abnormal (i.e. >5.5 mmol/L)?",
    "lab_hypoglycemia": "If a lab test for Hypoglycemia is ordered for this patient right now, \
    will it come back back abnormal (i.e. <3.9 mmol/L)?",
    "lab_hyponatremia": "If a lab test for Hyponatremia is ordered for this patient right now, \
    will it come back back abnormal (i.e. <135 mmol/L)?",
    "lab_thrombocytopenia": "If a lab test for Thrombocytopenia is ordered for this patient right now, \
    will it come back back abnormal? (i.e. <150 109/L)",
    "guo_los": "If this patient is admitted to the hospital right now, is the patient likely to have a \
    length of stay of at least 7 days (i.e. a week)?",
    "guo_readmission": "If this patient is discharged from the hospital right now, is the patient likely \
    to be readmitted to the hospital within 30 days?",
    "guo_icu": "If this patient is admitted to the hospital right now, is the patient likely to be \
    transferred to the ICU at any point during their admission?",
}


##################################
# Task Definitions
##################################
# Exact task definitions from the EHRSHOT paper
TASK_DEFS = {}

TASK_DEFS["guo_icu"] = (
    "Predict whether a patient will be transferred to the ICU during a visit to the "
    "hospital. The prediction time is at 11:59pm on the day of admission, and ICU "
    "transfers that occur on the same day as admission are ignored."
)

TASK_DEFS["guo_los"] = (
    "Predict whether a patient’s total length of stay during a visit to the hospital "
    "will be at least 7 days. The prediction time is at 11:59pm on the day of admission, "
    "and visits that last less than one day (i.e. discharge occurs on the same day of "
    "admission) are ignored."
)

TASK_DEFS["guo_readmission"] = (
    "Predict whether a patient will be re-admitted to the hospital within 30 days after "
    "being discharged from a visit. The prediction time is at 11:59pm on the day of "
    "admission, and admissions where a readmission occurs on the same day as the "
    "corresponding discharge are ignored."
)

# Use the binary classification formulation here

TASK_DEFS["lab_thrombocytopenia"] = (
    "Predict whether a thrombocytopenia lab comes back as normal (>=150 109/L) "
    " or abnormal (<150 109/L). We consider all lab results coded as LOINC/LP393218-5, "
    "LOINC/LG32892-8, or LOINC/777-3. The prediction time is immediately before the lab "
    "result is recorded."
)

TASK_DEFS["lab_hyperkalemia"] = (
    "Predict whether a hyperkalemia lab comes back as normal (<=5.5 mmol/L) or "
    "abnormal (>5.5 mmol/L). We consider all lab results coded as LOINC LG7931-1, "
    "LOINC/LP386618-5, LOINC/LG109906, LOINC/6298-4, or LOINC/2823-3. The prediction "
    "time is immediately before the lab result is recorded."
)

TASK_DEFS["lab_hypoglycemia"] = (
    "Predict whether a hypoglycemia lab comes back as normal (>=3.9 mmol/L) or "
    "abnormal (<3.9 mmol/L). We consider all lab results coded as SNOMED/33747003, "
    "LOINC/LP4161453, or LOINC/14749-6. The prediction time is immediately before the "
    "lab result is recorded."
)

TASK_DEFS["lab_hyponatremia"] = (
    "Predict whether a hyponatremia lab comes back as normal (>=135 mmol/L) or "
    "abnormal (<135 mmol/L). We consider all lab results coded as LOINC/LG11363-5, "
    "LOINC/2951-2, or LOINC/2947-0. The prediction time is immediately before the lab "
    "result is recorded."
)

TASK_DEFS["lab_anemia"] = (
    "Predict whether an anemia lab comes back as normal (>=120 g/L) or "
    "abnormal (<120 g/L). We consider all lab results coded as LOINC/LP392452-1. "
    "The prediction time is immediately before the lab result is recorded."
)


##################################
# Personas
##################################

# Generated from GPT4o using the wikipedia definition and this prompt:
#
# Read the following definition of a medical condition and suggest the most
# likely medical specialists (up to 5) who would diagnosis and treat a patient
# with this condition. Only list the title and respond with a Python List object.
#
# "{CLINICAL_DEFINIION}""

PERSONAS = {}

PERSONAS["new_acutemi"] = [
    "Cardiologist",
    "Emergency Medicine Physician",
    "Interventional Cardiologist",
    "Intensivist",
    "Primary Care Physician",
]

PERSONAS["new_celiac"] = [
    "Gastroenterologist",
    "Immunologist",
    "Endocrinologist",
    "Pediatrician",
    "Primary Care Physician",
]

PERSONAS["new_hyperlipidemia"] = [
    "Cardiologist",
    "Endocrinologist",
    "Primary Care Physician",
    "Lipidologist",
    "Gastroenterologist",
]

PERSONAS["new_hypertension"] = [
    "Primary Care Physician",
    "Cardiologist",
    "Nephrologist",
    "Endocrinologist",
    "Internist",
]

PERSONAS["new_lupus"] = [
    "Rheumatologist",
    "Immunologist",
    "Nephrologist",
    "Cardiologist",
    "Dermatologist",
]

PERSONAS["new_pancan"] = [
    "Oncologist",
    "Gastroenterologist",
    "Radiologist",
    "Hepatobiliary Surgeon",
    "Genetic Counselor",
]

# Generated from GPT4o using the wikipedia definition and this prompt:
#
# Read the following definition of a medical prediction task, and suggest
# the most likely medical specialists or professions (up to 5) who would be
# involved in either predicting, treating, or managing a patient who might
# have this event occur. Only list the title and respond with a Python List object.
#
# "{TASK_DEFINITION}""

PERSONAS["guo_icu"] = [
    "Intensivist",
    "Hospitalist",
    "Critical Care Nurse",
    "Emergency Medicine Physician",
    "Medical Data Scientist",
]

PERSONAS["guo_los"] = [
    "Hospitalist",
    "Internal Medicine Specialist",
    "Intensivist",
    "Discharge Planner",
    "Clinical Data Scientist",
]

PERSONAS["guo_readmission"] = [
    "Hospitalist",
    "Internal Medicine Specialist",
    "Case Manager",
    "Discharge Planner",
    "Primary Care Physician",
]

PERSONAS["lab_anemia"] = [
    "Hematologist",
    "Primary Care Physician",
    "Internal Medicine Specialist",
    "Clinical Pathologist",
    "Nurse Practitioner",
]

PERSONAS["lab_hyperkalemia"] = [
    "Nephrologist",
    "Endocrinologist",
    "Cardiologist",
    "Primary Care Physician",
    "Clinical Laboratory Scientist",
]

PERSONAS["lab_hypoglycemia"] = [
    "Endocrinologist",
    "Primary Care Physician",
    "Diabetologist",
    "Clinical Laboratory Scientist",
    "Nurse Practitioner",
]

PERSONAS["lab_hyponatremia"] = [
    "Nephrologist",
    "Endocrinologist",
    "Primary Care Physician",
    "Emergency Medicine Physician",
    "Clinical Laboratory Scientist",
]

PERSONAS["lab_thrombocytopenia"] = [
    "Hematologist",
    "Primary Care Physician",
    "Pathologist",
    "Oncologist",
    "Critical Care Specialist",
]

##################################
# Clinical Definitions
##################################

# Generated from GPT4o using the Wikipedia definition and this prompt:
#
# "Read the following text describing a clinical condition. Provide a short,
# diagnostic-focused definition that would enable a doctor LLM to review a
# patient's historical EHR and predict the likelihood of developing the
# condition in question: {text}"

CLINICAL_SHORT_DEFS = {}

CLINICAL_SHORT_DEFS["new_acutemi"] = (
    "A myocardial infarction (MI), commonly known as a heart attack, occurs when "
    "blood flow to the heart muscle is blocked, leading to tissue death. Key "
    "symptoms include chest pain radiating to the left shoulder, arm, or jaw, "
    "shortness of breath, nausea, and cold sweats. Atypical presentations, "
    "especially in women and the elderly, include neck pain, fatigue, and "
    "minimal symptoms. Major risk factors are coronary artery disease, high "
    "blood pressure, smoking, diabetes, and obesity. Diagnosis involves ECGs and "
    "blood tests for troponin levels. Immediate treatment aims to restore blood "
    "flow via percutaneous coronary intervention (PCI) or thrombolysis. Long-term "
    "management includes lifestyle changes and medications like aspirin and "
    "statins."
)

CLINICAL_SHORT_DEFS["new_celiac"] = (
    "Celiac disease is a chronic autoimmune disorder primarily affecting the small "
    "intestine, triggered by an intolerance to gluten (proteins found in wheat, "
    "rye, and barley). Symptoms range from gastrointestinal issues like chronic "
    "diarrhea and malabsorption to non-gastrointestinal symptoms or even no "
    "symptoms. It is linked with other autoimmune diseases such as Type 1 diabetes "
    "and Hashimoto's thyroiditis. Diagnosis involves blood antibody tests, "
    "intestinal biopsies, and genetic testing. The only effective treatment is a "
    "strict lifelong gluten-free diet, which mitigates symptoms and reduces "
    "complications."
)

CLINICAL_SHORT_DEFS["new_hyperlipidemia"] = (
    "Hyperlipidemia is characterized by abnormally high levels of lipids or "
    "lipoproteins in the blood, including fats, triglycerides, cholesterol, and "
    "phospholipids. It can result from genetic factors (primary hyperlipidemia) or "
    "underlying conditions like diabetes (secondary hyperlipidemia). This "
    "condition is a modifiable risk factor for cardiovascular disease and may also "
    "predispose individuals to acute pancreatitis. Diagnosis requires laboratory "
    "tests to measure lipid levels, and management often involves chronic "
    "medication to control these levels."
)

CLINICAL_SHORT_DEFS["new_hypertension"] = (
    "Hypertension, or high blood pressure, is a chronic medical condition "
    "characterized by persistently elevated blood pressure in the arteries, with a "
    "resting measurement at or above 130/80 mmHg. It is a significant risk factor "
    "for numerous cardiovascular and systemic diseases, including stroke, coronary "
    "artery disease, heart failure, atrial fibrillation, and chronic kidney "
    "disease. Hypertension is divided into primary (essential) hypertension, "
    "accounting for 90-95% of cases, which is due to nonspecific lifestyle and "
    "genetic factors, and secondary hypertension, due to identifiable causes like "
    "chronic kidney disease and endocrine disorders. Identifying hypertension in a "
    "patient's EHR involves reviewing blood pressure readings, assessing for risk "
    "factors such as obesity, high salt intake, and smoking, and noting any related "
    "health conditions or medications."
)

CLINICAL_SHORT_DEFS["new_lupus"] = (
    "Systemic lupus erythematosus (SLE) is an autoimmune disease where the body's "
    "immune system mistakenly attacks healthy tissues, causing inflammation and "
    "damage in various organs. Common symptoms include joint pain and swelling, "
    "fever, chest pain, hair loss, mouth ulcers, swollen lymph nodes, fatigue, and "
    "a distinctive facial rash. Diagnosis involves a combination of clinical "
    "symptoms and laboratory tests, particularly the presence of anti-nuclear "
    "antibodies. Risk factors include genetic predisposition, female sex hormones, "
    "sunlight exposure, smoking, vitamin D deficiency, and certain infections. "
    "There is no cure, but treatments such as NSAIDs, corticosteroids, "
    "immunosuppressants, hydroxychloroquine, and methotrexate can manage symptoms. "
    "SLE significantly increases the risk of cardiovascular disease and can "
    "complicate pregnancies."
)

CLINICAL_SHORT_DEFS["new_pancan"] = (
    "Pancreatic cancer, primarily pancreatic adenocarcinoma, arises from the "
    "pancreas and often goes undetected until advanced stages. Key diagnostic "
    "indicators include abdominal pain, jaundice, weight loss, and a history of "
    "smoking, obesity, diabetes, or certain genetic conditions. Screening relies "
    "on imaging, blood tests, and biopsies, with risk factors including smoking "
    "(25% of cases) and genetic predispositions (5-10%). Early diagnosis is rare, "
    "with over half of cases occurring in individuals over 70. Neuroendocrine "
    "tumors, though less common and aggressive, also originate in the pancreas. "
    "Treatment options vary by stage and include surgery, radiotherapy, and "
    "chemotherapy, but the prognosis remains poor, particularly for late-stage "
    "adenocarcinoma."
)

# Definitions are comprised of the first section of each task's Wikpedia page.
# Retreived July 19, 2024.
CLINICAL_DEFS = {}

CLINICAL_DEFS["new_acutemi"] = (
    "A myocardial infarction (MI), commonly known as a heart attack, occurs when "
    "blood flow decreases or stops in one of the coronary arteries of the heart, "
    "causing infarction (tissue death) to the heart muscle. The most common symptom "
    "is retrosternal chest pain or discomfort that classically radiates to the left "
    "shoulder, arm, or jaw. The pain may occasionally feel like heartburn. Other "
    "symptoms may include shortness of breath, nausea, feeling faint, a cold sweat, "
    "feeling tired, and decreased level of consciousness. About 30% of people have "
    "atypical symptoms. Women more often present without chest pain and instead have "
    "neck pain, arm pain or feel tired. Among those over 75 years old, about 5% have "
    "had an MI with little or no history of symptoms. An MI may cause heart failure, "
    "an irregular heartbeat, cardiogenic shock or cardiac arrest. Most MIs occur due "
    "to coronary artery disease. Risk factors include high blood pressure, smoking, "
    "diabetes, lack of exercise, obesity, high blood cholesterol, poor diet, and "
    "excessive alcohol intake. The complete blockage of a coronary artery caused by a "
    "rupture of an atherosclerotic plaque is usually the underlying mechanism of an MI. "
    "MIs are less commonly caused by coronary artery spasms, which may be due to "
    "cocaine, significant emotional stress (often known as Takotsubo syndrome or broken "
    "heart syndrome) and extreme cold, among others. Many tests are helpful to help "
    "with diagnosis, including electrocardiograms (ECGs), blood tests and coronary "
    "angiography. An ECG, which is a recording of the heart's electrical activity, may "
    "confirm an ST elevation MI (STEMI), if ST elevation is present. Commonly used blood "
    "tests include troponin and less often creatine kinase MB. Treatment of an MI is "
    "time-critical. Aspirin is an appropriate immediate treatment for a suspected MI. "
    "Nitroglycerin or opioids may be used to help with chest pain; however, they do not "
    "improve overall outcomes. Supplemental oxygen is recommended in those with low "
    "oxygen levels or shortness of breath. In a STEMI, treatments attempt to restore "
    "blood flow to the heart and include percutaneous coronary intervention (PCI), where "
    "the arteries are pushed open and may be stented, or thrombolysis, where the blockage "
    "is removed using medications. People who have a non-ST elevation myocardial infarction "
    "(NSTEMI) are often managed with the blood thinner heparin, with the additional use of "
    "PCI in those at high risk. In people with blockages of multiple coronary arteries and "
    "diabetes, coronary artery bypass surgery (CABG) may be recommended rather than "
    "angioplasty. After an MI, lifestyle modifications, along with long-term treatment with "
    "aspirin, beta blockers and statins, are typically recommended. Worldwide, about 15.9 "
    "million myocardial infarctions occurred in 2015. More than 3 million people had an ST "
    "elevation MI, and more than 4 million had an NSTEMI. STEMIs occur about twice as often "
    "in men as women. About one million people have an MI each year in the United States. In "
    "the developed world, the risk of death in those who have had a STEMI is about 10%. Rates "
    "of MI for a given age have decreased globally between 1990 and 2010. In 2011, an MI was "
    "one of the top five most expensive conditions during inpatient hospitalizations in the "
    "US, with a cost of about $11.5 billion for 612,000 hospital stays."
)

CLINICAL_DEFS["new_celiac"] = (
    "Coeliac disease (British English) or celiac disease (American English) is a "
    "long-term autoimmune disorder, primarily affecting the small intestine, "
    "where individuals develop intolerance to gluten, present in foods such as "
    "wheat, rye and barley. Classic symptoms include gastrointestinal problems "
    "such as chronic diarrhoea, abdominal distention, malabsorption, loss of "
    "appetite, and among children failure to grow normally. Non-classic symptoms "
    "are more common, especially in people older than two years. There may be "
    "mild or absent gastrointestinal symptoms, a wide number of symptoms involving "
    "any part of the body, or no obvious symptoms. Coeliac disease was first "
    "described in childhood; however, it may develop at any age. It i-s associated "
    "with other autoimmune diseases, such as Type 1 diabetes mellitus and Hashimoto's "
    "thyroiditis, among others. Coeliac disease is caused by a reaction to gluten, a "
    "group of various proteins found in wheat and in other grains such as barley and "
    "rye. Moderate quantities of oats, free of contamination with other gluten-"
    "containing grains, are usually tolerated. The occurrence of problems may depend "
    "on the variety of oat. It occurs more often in people who are genetically "
    "predisposed. Upon exposure to gluten, an abnormal immune response may lead to the "
    "production of several different autoantibodies that can affect a number of "
    "different organs. In the small bowel, this causes an inflammatory reaction and "
    "may produce shortening of the villi lining the small intestine (villous atrophy). "
    "This affects the absorption of nutrients, frequently leading to anaemia. Diagnosis "
    "is typically made by a combination of blood antibody tests and intestinal biopsies, "
    "helped by specific genetic testing. Making the diagnosis is not always "
    "straightforward. About 10% of the time, the autoantibodies in the blood are "
    "negative, and many people have only minor intestinal changes with normal villi. "
    "People may have severe symptoms and they may be investigated for years before a "
    "diagnosis is achieved. As a result of screening, the diagnosis is increasingly "
    "being made in people who have no symptoms. Evidence regarding the effects of "
    "screening, however, is not sufficient to determine its usefulness. While the "
    "disease is caused by a permanent intolerance to gluten proteins, it is distinct "
    "from wheat allergy, which is much more rare. The only known effective treatment is "
    "a strict lifelong gluten-free diet, which leads to recovery of the intestinal "
    "lining (mucous membrane), improves symptoms, and reduces the risk of developing "
    "complications in most people. If untreated, it may result in cancers such as "
    "intestinal lymphoma, and a slightly increased risk of early death. Rates vary "
    "between different regions of the world, from as few as 1 in 300 to as many as 1 in "
    "40, with an average of between 1 in 100 and 1 in 170 people. It is estimated that "
    "80% of cases remain undiagnosed, usually because of minimal or absent "
    "gastrointestinal complaints and lack of knowledge of symptoms and diagnostic "
    "criteria. Coeliac disease is slightly more common in women than in men."
)

CLINICAL_DEFS["new_hyperlipidemia"] = (
    "Hyperlipidemia is abnormally high levels of any or all lipids (e.g. fats, "
    "triglycerides, cholesterol, phospholipids) or lipoproteins in the blood. "
    "The term hyperlipidemia refers to the laboratory finding itself and is also "
    "used as an umbrella term covering any of various acquired or genetic "
    "disorders that result in that finding. Hyperlipidemia represents a subset "
    "of dyslipidemia and a superset of hypercholesterolemia. Hyperlipidemia is "
    "usually chronic and requires ongoing medication to control blood lipid "
    "levels. Lipids (water-insoluble molecules) are transported in a protein "
    "capsule. The size of that capsule, or lipoprotein, determines its density. "
    "The lipoprotein density and type of apolipoproteins it contains determines "
    "the fate of the particle and its influence on metabolism. Hyperlipidemias "
    "are divided into primary and secondary subtypes. Primary hyperlipidemia is "
    "usually due to genetic causes (such as a mutation in a receptor protein), "
    "while secondary hyperlipidemia arises due to other underlying causes such "
    "as diabetes. Lipid and lipoprotein abnormalities are common in the general "
    "population and are regarded as modifiable risk factors for cardiovascular "
    "disease due to their influence on atherosclerosis. In addition, some forms "
    "may predispose to acute pancreatitis."
)

CLINICAL_DEFS["new_hypertension"] = (
    "Hypertension, also known as high blood pressure, is a long-term medical "
    "condition in which the blood pressure in the arteries is persistently "
    "elevated. High blood pressure usually does not cause symptoms itself. It is, "
    "however, a major risk factor for stroke, coronary artery disease, heart "
    "failure, atrial fibrillation, peripheral arterial disease, vision loss, "
    "chronic kidney disease, and dementia. Hypertension is a major cause of "
    "premature death worldwide. High blood pressure is classified as primary "
    "(essential) hypertension or secondary hypertension. About 90-95% of cases are "
    "primary, defined as high blood pressure due to nonspecific lifestyle and "
    "genetic factors. Lifestyle factors that increase the risk include excess salt "
    "in the diet, excess body weight, smoking, physical inactivity and alcohol use. "
    "The remaining 5-10% of cases are categorized as secondary hypertension, defined "
    "as high blood pressure due to a clearly identifiable cause, such as chronic "
    "kidney disease, narrowing of the kidney arteries, an endocrine disorder, or the "
    "use of birth control pills. Blood pressure is classified by two measurements, "
    "the systolic (first number) and diastolic (second number) pressures. For most "
    "adults, normal blood pressure at rest is within the range of 100-140 millimeters "
    "mercury (mmHg) systolic and 60-90 mmHg diastolic. For most adults, high blood "
    "pressure is present if the resting blood pressure is persistently at or above "
    "130/80 or 140/90 mmHg. Different numbers apply to children. Ambulatory blood "
    "pressure monitoring over a 24-hour period appears more accurate than office-based "
    "blood pressure measurement. Lifestyle changes and medications can lower blood "
    "pressure and decrease the risk of health complications. Lifestyle changes include "
    "weight loss, physical exercise, decreased salt intake, reducing alcohol intake, "
    "and a healthy diet. If lifestyle changes are not sufficient, blood pressure "
    "medications are used. Up to three medications taken concurrently can control "
    "blood pressure in 90% of people. The treatment of moderately high arterial blood "
    "pressure (defined as >160/100 mmHg) with medications is associated with an improved "
    "life expectancy. The effect of treatment of blood pressure between 130/80 mmHg and "
    "160/100 mmHg is less clear, with some reviews finding benefit and others finding "
    "unclear benefit. High blood pressure affects 33% of the population globally. About "
    "half of all people with high blood pressure do not know that they have it. In 2019, "
    "high blood pressure was believed to have been a factor in 19% of all deaths (10.4 "
    "million globally)."
)

CLINICAL_DEFS["new_lupus"] = (
    "Lupus, technically known as systemic lupus erythematosus (SLE), is an autoimmune "
    "disease in which the body's immune system mistakenly attacks healthy tissue in many "
    "parts of the body. Symptoms vary among people and may be mild to severe. Common "
    "symptoms include painful and swollen joints, fever, chest pain, hair loss, mouth "
    "ulcers, swollen lymph nodes, feeling tired, and a red rash which is most commonly on "
    "the face. Often there are periods of illness, called flares, and periods of remission "
    "during which there are few symptoms. The cause of SLE is not clear. It is thought to "
    "involve a combination of genetics and environmental factors. Among identical twins, "
    "if one is affected there is a 24% chance the other one will also develop the disease. "
    "Female sex hormones, sunlight, smoking, vitamin D deficiency, and certain infections "
    "are also believed to increase a person's risk. The mechanism involves an immune "
    "response by autoantibodies against a person's own tissues. These are most commonly "
    "anti-nuclear antibodies and they result in inflammation. Diagnosis can be difficult "
    "and is based on a combination of symptoms and laboratory tests. There are a number of "
    "other kinds of lupus erythematosus including discoid lupus erythematosus, neonatal "
    "lupus, and subacute cutaneous lupus erythematosus. There is no cure for SLE, but there "
    "are experimental and symptomatic treatments. Treatments may include NSAIDs, "
    "corticosteroids, immunosuppressants, hydroxychloroquine, and methotrexate. Although "
    "corticosteroids are rapidly effective, long-term use results in side effects. "
    "Alternative medicine has not been shown to affect the disease. Men have higher "
    "mortality. SLE significantly increases the risk of cardiovascular disease, with this "
    "being the most common cause of death. While women with lupus have higher risk "
    "pregnancies, most are successful. Rate of SLE varies between countries from 20 to 70 "
    "per 100,000. Women of childbearing age are affected about nine times more often than "
    "men. While it most commonly begins between the ages of 15 and 45, a wide range of ages "
    "can be affected. Those of African, Caribbean, and Chinese descent are at higher risk "
    "than those of European descent. Rates of disease in the developing world are unclear. "
    "Lupus is Latin for 'wolf': the disease was so-named in the 13th century as the rash "
    "was thought to appear like a wolf's bite."
)

CLINICAL_DEFS["new_pancan"] = (
    "Pancreatic cancer arises when cells in the pancreas, a glandular organ behind "
    "the stomach, begin to multiply out of control and form a mass. These cancerous "
    "cells have the ability to invade other parts of the body. A number of types of "
    "pancreatic cancer are known. The most common, pancreatic adenocarcinoma, accounts "
    "for about 90% of cases, and the term 'pancreatic cancer' is sometimes used to "
    "refer only to that type. These adenocarcinomas start within the part of the "
    "pancreas that makes digestive enzymes. Several other types of cancer, which "
    "collectively represent the majority of the non-adenocarcinomas, can also arise "
    "from these cells. About 1-2% of cases of pancreatic cancer are neuroendocrine "
    "tumors, which arise from the hormone-producing cells of the pancreas. These are "
    "generally less aggressive than pancreatic adenocarcinoma. Signs and symptoms of "
    "the most-common form of pancreatic cancer may include yellow skin, abdominal or "
    "back pain, unexplained weight loss, light-colored stools, dark urine, and loss of "
    "appetite. Usually, no symptoms are seen in the disease's early stages, and "
    "symptoms that are specific enough to suggest pancreatic cancer typically do not "
    "develop until the disease has reached an advanced stage. By the time of diagnosis, "
    "pancreatic cancer has often spread to other parts of the body. Pancreatic cancer "
    "rarely occurs before the age of 40, and more than half of cases of pancreatic "
    "adenocarcinoma occur in those over 70. Risk factors for pancreatic cancer include "
    "tobacco smoking, obesity, diabetes, and certain rare genetic conditions. About 25% "
    "of cases are linked to smoking, and 5-10% are linked to inherited genes. Pancreatic "
    "cancer is usually diagnosed by a combination of medical imaging techniques such as "
    "ultrasound or computed tomography, blood tests, and examination of tissue samples "
    "(biopsy). The disease is divided into stages, from early (stage I) to late (stage "
    "IV). Screening the general population has not been found to be effective. The risk "
    "of developing pancreatic cancer is lower among non-smokers, and people who maintain "
    "a healthy weight and limit their consumption of red or processed meat; the risk is "
    "greater for men, smokers, and those with diabetes. There is some evidence that links "
    "high levels of red meat consumption to increased risk of pancreatic cancer. Smokers' "
    "risk of developing the disease decreases immediately upon quitting, and almost "
    "returns to that of the rest of the population after 20 years. Pancreatic cancer can "
    "be treated with surgery, radiotherapy, chemotherapy, palliative care, or a "
    "combination of these. Treatment options are partly based on the cancer stage. "
    "Surgery is the only treatment that can cure pancreatic adenocarcinoma, and may also "
    "be done to improve quality of life without the potential for cure. Pain management "
    "and medications to improve digestion are sometimes needed. Early palliative care is "
    "recommended even for those receiving treatment that aims for a cure. Pancreatic "
    "cancer is among the most deadly forms of cancer globally, with one of the lowest "
    "survival rates. In 2015, pancreatic cancers of all types resulted in 411,600 deaths "
    "globally. Pancreatic cancer is the fifth-most-common cause of death from cancer in "
    "the United Kingdom, and the third most-common in the United States. The disease "
    "occurs most often in the developed world, where about 70% of the new cases in 2012 "
    "originated. Pancreatic adenocarcinoma typically has a very poor prognosis; after "
    "diagnosis, 25% of people survive one year and 12% live for five years. For cancers "
    "diagnosed early, the five-year survival rate rises to about 20%. Neuroendocrine "
    "cancers have better outcomes; at five years from diagnosis, 65% of those diagnosed "
    "are living, though survival considerably varies depending on the type of tumor."
)


##################################
# Coded Definitions
##################################

# generated from the OMOP Athena vocabulary
# phenotype definition: parent->descendants

CODE_DEFS = {
    "new_acutemi": {
        "SNOMED/57054005": {
            "descendants": {
                "ICD9CM/410.80",
                "SNOMED/1204155000",
                "ICD10CM/I21.4",
                "SNOMED/401314000",
                "ICD9CM/410.1",
                "SNOMED/17531000119105",
                "ICD9CM/410.90",
                "SNOMED/836294006",
                "SNOMED/703251009",
                "ICD9CM/410.31",
                "SNOMED/12238151000119107",
                "SNOMED/44011000000104",
                "SNOMED/73795002",
                "SNOMED/836293000",
                "SNOMED/83351000000106",
                "SNOMED/23311000119105",
                "ICD9CM/410.10",
                "SNOMED/285991000119100",
                "SNOMED/44821000087100",
                "SNOMED/868226001",
                "SNOMED/44841000087109",
                "SNOMED/1208872002",
                "SNOMED/703213009",
                "SNOMED/194803008",
                "SNOMED/285981000119103",
                "SNOMED/122701000000102",
                "ICD9CM/410.2",
                "SNOMED/195545006",
                "SNOMED/194808004",
                "SNOMED/72977004",
                "SNOMED/282006",
                "ICD10CM/I21.0",
                "SNOMED/401303003",
                "ICD10CM/I21.11",
                "SNOMED/15712881000119105",
                "ICD9CM/410.22",
                "ICD9CM/410.51",
                "SNOMED/233834004",
                "SNOMED/45881000000101",
                "SNOMED/840316004",
                "ICD9CM/410.3",
                "SNOMED/15713201000119105",
                "SNOMED/471851000000100",
                "SNOMED/70211005",
                "SNOMED/471711000000109",
                "SNOMED/54329005",
                "HCPCS/G8009",
                "SNOMED/194811003",
                "SNOMED/233828006",
                "SNOMED/44811000087108",
                "SNOMED/76593002",
                "ICD9CM/410.71",
                "SNOMED/15712921000119103",
                "SNOMED/12238111000119106",
                "SNOMED/233837006",
                "SNOMED/44001000000101",
                "SNOMED/233830008",
                "SNOMED/194798004",
                "SNOMED/840609007",
                "SNOMED/703164000",
                "ICD9CM/410.91",
                "ICD9CM/410.70",
                "SNOMED/703165004",
                "SNOMED/70422006",
                "HCPCS/G8007",
                "SNOMED/194807009",
                "ICD9CM/410.52",
                "ICD10CM/I21.3",
                "SNOMED/233836002",
                "ICD9CM/410.4",
                "SNOMED/58612006",
                "SNOMED/583001000000107",
                "ICD9CM/410.11",
                "SNOMED/57054005",
                "SNOMED/846668006",
                "ICD9CM/410",
                "SNOMED/1204151009",
                "ICD9CM/410.02",
                "SNOMED/233832000",
                "SNOMED/233829003",
                "SNOMED/44851000087107",
                "SNOMED/15962541000119106",
                "SNOMED/233827001",
                "SNOMED/64627002",
                "SNOMED/307140009",
                "SNOMED/194810002",
                "SNOMED/194805001",
                "SNOMED/79009004",
                "SNOMED/155321002",
                "ICD9CM/410.40",
                "ICD9CM/410.32",
                "SNOMED/1204154001",
                "SNOMED/840309000",
                "SNOMED/233835003",
                "ICD10CM/I21.2",
                "SNOMED/233831007",
                "SNOMED/59063002",
                "ICD9CM/410.30",
                "ICD10CM/I21.1",
                "ICD9CM/410.72",
                "ICD9CM/410.20",
                "ICD10CM/I21.01",
                "ICD9CM/410.5",
                "SNOMED/15990001",
                "ICD10CM/I21.21",
                "ICD9CM/410.81",
                "SNOMED/703212004",
                "SNOMED/46001000000109",
                "SNOMED/896689003",
                "HCPCS/G8010",
                "SNOMED/1163440003",
                "ICD9CM/410.00",
                "ICD10CM/I21.9",
                "ICD9CM/410.0",
                "ICD9CM/410.01",
                "SNOMED/896691006",
                "SNOMED/868214006",
                "SNOMED/868224003",
                "SNOMED/15963181000119104",
                "ICD9CM/410.41",
                "SNOMED/623341000000106",
                "ICD9CM/410.42",
                "SNOMED/233825009",
                "ICD9CM/410.12",
                "SNOMED/868220007",
                "SNOMED/868225002",
                "ICD9CM/410.82",
                "SNOMED/52035003",
                "ICD9CM/410.21",
                "HCPCS/G8006",
                "SNOMED/62695002",
                "SNOMED/703252002",
                "SNOMED/15713161000119100",
                "SNOMED/868217004",
                "SNOMED/155319007",
                "SNOMED/304914007",
                "SNOMED/15713041000119103",
                "SNOMED/44831000087103",
                "ICD9CM/410.8",
                "SNOMED/840312002",
                "ICD9CM/410.50",
                "SNOMED/15712841000119100",
                "SNOMED/1204222000",
                "SNOMED/194809007",
                "ICD10CM/I21.19",
                "ICD9CM/410.92",
                "SNOMED/65547006",
                "ICD10CM/I21",
                "SNOMED/1089451000000100",
                "ICD9CM/410.7",
                "SNOMED/840680009",
                "ICD10CM/I21.29",
                "SNOMED/846683001",
                "ICD10CM/I21.02",
                "SNOMED/896696001",
                "SNOMED/233826005",
                "SNOMED/1204152002",
                "SNOMED/15713001000119100",
                "SNOMED/836295007",
                "SNOMED/703253007",
                "SNOMED/15713121000119105",
                "SNOMED/15713081000119108",
                "ICD10CM/I21.09",
                "SNOMED/70998009",
                "SNOMED/896697005",
                "SNOMED/15712961000119108",
                "ICD9CM/410.9",
                "SNOMED/233838001",
                "SNOMED/1089471000000109",
                "SNOMED/233833005",
                "SNOMED/412771006",
            }
        }
    },
    "new_celiac": {
        "SNOMED/396331005": {
            "descendants": {
                "SNOMED/197481005",
                "SNOMED/45259000",
                "SNOMED/91867008",
                "SNOMED/197479008",
                "ICD9CM/579.0",
                "SNOMED/266478000",
                "SNOMED/155842007",
                "ICD10CM/K90.0",
                "SNOMED/770593004",
                "SNOMED/396331005",
                "SNOMED/197478000",
                "SNOMED/61715008",
                "SNOMED/23829007",
                "SNOMED/1197730009",
                "SNOMED/396330006",
                "SNOMED/722386009",
            }
        }
    },
    "new_hyperlipidemia": {
        "SNOMED/55822004": {
            "descendants": {
                "SNOMED/299465007",
                "ICD10CM/E78.2",
                "SNOMED/214021000000106",
                "SNOMED/15771000119109",
                "SNOMED/518631000000104",
                "SNOMED/426161002",
                "SNOMED/403830007",
                "SNOMED/154743001",
                "SNOMED/402473001",
                "SNOMED/238040008",
                "SNOMED/701000119103",
                "SNOMED/773649005",
                "ICD9CM/272.1",
                "SNOMED/302870006",
                "SNOMED/238080004",
                "SNOMED/389985001",
                "SNOMED/137931000119102",
                "ICD10CM/E78.00",
                "SNOMED/238079002",
                "SNOMED/238085009",
                "SNOMED/129590000",
                "SNOMED/13644009",
                "SNOMED/238083002",
                "SNOMED/34171006",
                "SNOMED/403829002",
                "SNOMED/238078005",
                "SNOMED/238081000",
                "SNOMED/238039006",
                "SNOMED/403831006",
                "SNOMED/397915002",
                "SNOMED/238084008",
                "SNOMED/267433009",
                "SNOMED/778111000000106",
                "ICD10CM/E78.01",
                "SNOMED/767133009",
                "SNOMED/267432004",
                "SNOMED/402786009",
                "ICD9CM/272.2",
                "SNOMED/129589009",
                "SNOMED/238076009",
                "SNOMED/275598004",
                "SNOMED/398036000",
                "SNOMED/190782002",
                "SNOMED/518591000000104",
                "SNOMED/55822004",
                "SNOMED/1571000119104",
                "SNOMED/402725005",
                "SNOMED/34528009",
                "SNOMED/238082007",
                "SNOMED/114831000119107",
                "SNOMED/190774002",
                "SNOMED/238077000",
                "ICD10CM/E78.5",
                "SNOMED/403827000",
                "SNOMED/778121000000100",
                "SNOMED/48190005",
                "SNOMED/238087001",
                "SNOMED/154741004",
                "SNOMED/238088006",
                "SNOMED/267434003",
                "SNOMED/633291000000106",
                "ICD10CM/E78.3",
                "SNOMED/34349009",
                "OMOP Extension/OMOP5166017",
                "SNOMED/402475008",
                "SNOMED/402785008",
                "SNOMED/267435002",
                "OMOP Extension/OMOP5181809",
                "SNOMED/238086005",
                "SNOMED/137941000119106",
                "ICD9CM/272.3",
                "SNOMED/403828005",
                "SNOMED/129591001",
                "SNOMED/154742006",
                "SNOMED/633301000000105",
                "SNOMED/190778004",
                "SNOMED/518601000000105",
                "ICD9CM/272.4",
                "SNOMED/1208738002",
                "SNOMED/190775001",
                "ICD10CM/E78.1",
                "SNOMED/445261005",
                "ICD9CM/272.0",
                "SNOMED/890601000000107",
                "ICD10CM/E78.4",
                "SNOMED/402726006",
                "SNOMED/1197489003",
                "SNOMED/773726000",
                "SNOMED/402787000",
                "SNOMED/33513003",
                "SNOMED/402727002",
                "ICD10CM/E78.49",
                "SNOMED/402474007",
                "ICD10CM/E78.0",
                "SNOMED/190777009",
                "SNOMED/31654005",
                "SNOMED/238089003",
                "SNOMED/491251000000107",
            }
        }
    },
    "new_hypertension": {
        "SNOMED/59621000": {
            "descendants": {
                "SNOMED/46481004",
                "SNOMED/72022006",
                "ICD9CM/401",
                "SNOMED/1201005",
                "ICD10CM/I10",
                "SNOMED/23717007",
                "SNOMED/266228004",
                "SNOMED/71874008",
                "SNOMED/9901000",
                "SNOMED/371125006",
                "SNOMED/194760004",
                "SNOMED/40511000119107",
                "SNOMED/63287004",
                "SNOMED/18416000",
                "ICD9CM/401.9",
                "SNOMED/1218009",
                "SNOMED/78975002",
                "SNOMED/78808002",
                "SNOMED/35303009",
                "SNOMED/155296003",
                "ICD9CM/401.1",
                "SNOMED/194758001",
                "ICD9CM/401.0",
                "SNOMED/19769006",
                "SNOMED/429457004",
                "SNOMED/59621000",
            }
        }
    },
    "new_lupus": {
        "SNOMED/55464009": {
            "descendants": {
                "SNOMED/156450004",
                "SNOMED/76521009",
                "SNOMED/239888002",
                "SNOMED/36402006",
                "SNOMED/403487009",
                "ICD10CM/M32.1",
                "SNOMED/295111000119108",
                "SNOMED/25380002",
                "ICD10CM/M32.0",
                "SNOMED/196138005",
                "SNOMED/203784000",
                "SNOMED/698694005",
                "SNOMED/309762007",
                "SNOMED/724781003",
                "ICD10CM/M32.8",
                "SNOMED/201435004",
                "SNOMED/11013005",
                "SNOMED/201436003",
                "SNOMED/403488004",
                "ICD10CM/M32.10",
                "ICD10CM/M32.12",
                "SNOMED/95644001",
                "ICD10CM/M32.14",
                "SNOMED/77753005",
                "SNOMED/95408003",
                "SNOMED/239889005",
                "SNOMED/52042003",
                "SNOMED/239886003",
                "SNOMED/68815009",
                "SNOMED/73286009",
                "SNOMED/773333003",
                "SNOMED/593481000000109",
                "SNOMED/409421000000100",
                "SNOMED/54072008",
                "SNOMED/239890001",
                "SNOMED/295101000119105",
                "ICD10CM/M32.13",
                "SNOMED/4676006",
                "ICD10CM/M32.19",
                "SNOMED/758321000000100",
                "SNOMED/239887007",
                "SNOMED/197608009",
                "SNOMED/201438002",
                "ICD10CM/M32.9",
                "SNOMED/201437007",
                "SNOMED/19682006",
                "SNOMED/55464009",
                "OMOP Extension/OMOP5166128",
                "ICD9CM/710.0",
                "ICD10CM/M32",
                "SNOMED/201439005",
                "SNOMED/295121000119101",
                "SNOMED/403486000",
            }
        }
    },
    "new_pancan": {
        "SNOMED/372003004": {
            "descendants": {
                "SNOMED/1268532006",
                "ICD9CM/157.2",
                "SNOMED/733351008",
                "ICD9CM/157.3",
                "ICD9CM/157.9",
                "ICD10CM/C25",
                "SNOMED/1268911008",
                "SNOMED/93715005",
                "SNOMED/1197286008",
                "SNOMED/16823941000119108",
                "ICD10CM/C25.4",
                "SNOMED/1268698000",
                "SNOMED/352701000119102",
                "ICD10CM/C24.1",
                "SNOMED/1197279000",
                "SNOMED/1259747009",
                "SNOMED/1259310006",
                "SNOMED/94082003",
                "ICD10CM/C25.9",
                "SNOMED/1259800005",
                "SNOMED/314999005",
                "SNOMED/1259539007",
                "SNOMED/735735001",
                "SNOMED/681721000119103",
                "SNOMED/1259682002",
                "SNOMED/1197283000",
                "ICD9CM/157.8",
                "SNOMED/371967001",
                "SNOMED/681911000119108",
                "ICD9CM/157.0",
                "SNOMED/109849001",
                "ICD10CM/C25.2",
                "ICD9CM/157.4",
                "SNOMED/1268561007",
                "SNOMED/1651000119109",
                "SNOMED/93668007",
                "SNOMED/1259309001",
                "SNOMED/1268546006",
                "SNOMED/93938001",
                "ICD9CM/156.2",
                "SNOMED/681971000119100",
                "ICD9CM/157",
                "ICD10CM/C25.7",
                "SNOMED/1268563005",
                "SNOMED/143391000119109",
                "SNOMED/93823001",
                "SNOMED/681621000119105",
                "SNOMED/721718003",
                "ICD9CM/157.1",
                "SNOMED/93939009",
                "ICD10CM/C25.3",
                "SNOMED/1259311005",
                "SNOMED/1259700000",
                "ICD10CM/C25.1",
                "ICD10CM/C25.0",
                "SNOMED/93843007",
                "SNOMED/1259358002",
                "SNOMED/372003004",
                "SNOMED/372119009",
                "SNOMED/681831000119107",
                "SNOMED/1259415000",
                "SNOMED/1259799006",
                "SNOMED/1268542008",
            }
        }
    },
}

##################################
# Action
##################################

ACTION_TMPL = (
    "Review the patient's EHR history. Based on all available medical evidence in the provided EHR, please "
    "answer the question: {question}"
)

ACTION_COT_TMPL = (
    "Review the patient's EHR history. Based on all available medical evidence in the provided EHR, please "
    "answer the question: {question} Reflect on the problem and generate a numbered list of the steps "
    "you took to reach your conclusion."
)


def get_task_config(task_name: str) -> Mapping[str, Any]:
    if task_name.startswith("guo"):
        return CONFIG["guo"]
    elif task_name.startswith("lab"):
        return CONFIG["lab"]
    else:
        return CONFIG["new"]


def lumia_prompt(
    task_name: str, config: Dict[str, Any], examples: List[Dict[str, Any]], timeline: Dict[str, Any]
) -> str:
    # Base template
    task_config: Mapping[str, Any] = get_task_config(task_name)
    tmpl: Dict[str, Any]
    tmpl = base_prompt(task_name, **task_config)

    # EHR => String converter
    if config.get("ehr_converter") == "codes_and_timestamps":
        ehr_converter = codes_and_timestamps
    elif config.get("ehr_converter") == "codes_only":
        ehr_converter = codes_only
    else:
        raise ValueError(f"Invalid `ehr_converter` strategy: {config.get('ehr_converter')}")

    # Examples
    # examples: str = "\n\n# Examples\n\n" + "\n".join([tmpl['example'].format(ehr=ehr_converter(e['ehr']), \
    # label=e['label']) for e in examples]) + "\n\n" if len(examples) > 0 else ""
    tmpl["instruction"] = tmpl["instruction"].replace(
        'Then respond with "yes" or "no" as your final output',
        'Then respond with "A" for yes or "B" for no as your final output',
    )
    # Prompt
    prompt: str = f"""
# Instructions
{tmpl['instruction']}

# Your Task
{tmpl['example'].format(ehr=ehr_converter(timeline['ehr']))}"""
    return prompt


def get_code_def(task_name: str):
    """Create a text list of all codes defining this task definition."""
    code_def: List[str] = []
    for parent in CODE_DEFS[task_name]:
        code_def.extend(CODE_DEFS[task_name][parent]["descendants"])

    return ", ".join(sorted(code_def))


def base_prompt(
    task_name: str,
    is_include_persona: bool = True,
    is_include_clinical_def: bool = True,
    is_include_code_def: bool = True,
    is_use_short_clinical_def: bool = True,
    is_include_cot: bool = True,
    seed: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build the base prompt template for the provided task_name. This will create
    a prompt that can be populated by specific labeled patient examples.

    Example output:
        'instruction': 'You are an expert endocrinologist at Stanford Healthcare, an academic medical center
                        affiliated with Stanford University. You specialize in diagnosing and treating
                        hypertension.\n\n Clinical Definition: Hypertension, or high blood pressure, is a
                        chronic medical condition characterized by persistently elevated blood pressure in
                        the arteries, with a resting measurement at or above 130/80 mmHg.
                        It is a significant risk factor for numerous cardiovascular and systemic diseases,
                        including stroke, coronary artery disease, heart failure, atrial fibrillation, and
                        chronic kidney disease. Hypertension is divided into primary (essential) hypertension,
                        accounting for 90-95% of cases, which is due to nonspecific lifestyle and
                        genetic factors, and secondary hypertension, due to identifiable causes like chronic
                        kidney disease and endocrine disorders. Identifying hypertension in a patient\'s EHR
                        involves reviewing blood pressure readings, assessing for risk factors such as obesity,
                        high salt intake, and smoking, and noting any related health conditions or medications.\n\n
                        Medical Code Definition: In an electronic health record (EHR), hypertension is denoted by the
                        occurrence of any of the following medical codes: ICD10CM/I10, ICD9CM/401, ICD9CM/401.0,
                        ICD9CM/401.1, ICD9CM/401.9, SNOMED/1201005, SNOMED/1218009, SNOMED/155296003,
                        SNOMED/18416000, SNOMED/194758001, SNOMED/194760004, SNOMED/19769006, SNOMED/23717007,
                        SNOMED/266228004, SNOMED/35303009, SNOMED/371125006, SNOMED/40511000119107, SNOMED/429457004,
                        SNOMED/46481004, SNOMED/59621000, SNOMED/63287004, SNOMED/71874008,
                        SNOMED/72022006, SNOMED/78808002, SNOMED/78975002, SNOMED/9901000\n\n
                        Instruction: Review the patient\'s EHR history and answer the following question:
                        Based on all available medical evidence in the provided EHR, is this patient likely to
                        receive a first time diagnosis of hypertension within the next year?'
        'example':     'Patient EHR: {ehr}\n\n'
        'delimiter':    '\n##\n'
    """
    prompt = []
    task_full_name = TASK_FULL_NAMES[task_name].lower()

    # 1. Persona
    if is_include_persona:
        persona = PERSONAS[task_name][0]
        prompt.append(
            "You are an expert {role} at Stanford Healthcare, an academic medical center "
            "affiliated with Stanford University. You specialize in predicting "
            "{task_full_name}.".format(
                role=persona.lower(),
                task_full_name=task_full_name,
            )
        )

    # 2. Natural language clinical definition
    if is_include_clinical_def and is_use_short_clinical_def:
        prompt.append("Clinical Definition: " + CLINICAL_SHORT_DEFS[task_name])
    elif is_include_clinical_def:
        prompt.append("Clinical Definition: " + CLINICAL_DEFS[task_name])

    # 3. Medical ontology phenotype definition (code-based inclusion criteria)
    if is_include_code_def:
        code_def = (
            "Medical Code Definition: In an electronic health record (EHR), {task_full_name} is denoted "
            "by the occurrence of any of the following medical codes: {code_def}"
        )
        prompt.append(
            code_def.format(
                task_full_name=task_full_name,
                code_def=get_code_def(task_name),
            )
        )

    # 4. Question for this task
    question: str = TASK_QUESTIONS[task_name]

    # 4. Action
    if is_include_cot:
        prompt.append("Instruction: " + ACTION_COT_TMPL.format(question=question))
    else:
        prompt.append("Instruction: " + ACTION_TMPL.format(question=question))

    # padding
    prompt = [item + "\n" for item in prompt]
    prompt_text = "\n".join(prompt)
    return {
        "instruction": prompt_text,
        "example": "Patient EHR:\n{ehr}\n\n",
        "delimiter": "\n##\n",
    }


def codes_and_timestamps(events: List[Dict[str, Any]]) -> str:
    """Format a list of MEDS events into a string.

    Example:
        > events = [
            { 'time' : datetime.datetime(2024, 1, 1), 'code' : 'ICD10CM/C25.9' },
            { 'time' : datetime.datetime(2024, 1, 2), 'code' : 'ICD10CM/C25.9' },
            { 'time' : datetime.datetime(2024, 1, 3), 'code' : 'ICD10CM/C25.9' },
        ]
        > format_ehr(events)
        # Output: "
        #   - 2024-01-01 ICD10CM/C25.9
        #   - 2024-01-02 ICD10CM/C25.9
        #   - 2024-01-03 ICD10CM/C25.9
        # "
    """
    return "\n".join([f"- {event['time'].strftime('%Y-%m-%d')} {event['code']}" for event in events])


def codes_only(events: List[Any]) -> str:
    """Format a list of MEDS events into a string.

    Example:
        events = [
            { 'time' : datetime.datetime(2024, 1, 1), 'code' : 'ICD10CM/C25.9' },
            { 'time' : datetime.datetime(2024, 1, 2), 'code' : 'ICD10CM/C25.9' },
            { 'time' : datetime.datetime(2024, 1, 3), 'code' : 'ICD10CM/C25.9' },
        ] or
        events = [
            'ICD10CM/C25.9',
            'ICD10CM/C25.9',
            'ICD10CM/C25.9',
        ]
        format_ehr(events)
        # Output: "
        #   - ICD10CM/C25.9
        #   - ICD10CM/C25.9
        #   - ICD10CM/C25.9
        # "
    """
    return "\n".join([f"- {event['code'] if isinstance(event, dict) else event}" for event in events])


def _process_prior_events_chunk(
    chunk_data: pd.DataFrame, grouped_a: pd.DataFrame, is_show_tqdm: bool = False
) -> List[List[str]]:
    """Process a chunk of label rows and return their prior events."""
    chunk_prior_events: List[List[str]] = []
    for _, row_b in tqdm(
        chunk_data.iterrows(), total=len(chunk_data), desc="Processing labels", disable=not is_show_tqdm
    ):
        # Find events for this patient that occur before the specified time
        patient_events = (
            grouped_a.get_group(row_b["subject_id"]) if row_b["subject_id"] in grouped_a.groups else pd.DataFrame()
        )

        if len(patient_events) == 0:
            chunk_prior_events.append([])
            continue

        # Filter events before the specified time
        prior_patient_events = patient_events[patient_events["time"] <= row_b["prediction_time"]]

        # Collect codes
        chunk_prior_events.append(list(prior_patient_events["code"]))

    return chunk_prior_events


def get_prior_events(df_data: pd.DataFrame, df_labels: pd.DataFrame, n_procs: int = 4) -> List[List[str]]:
    """
    Find events for each patient in `df_data` that occur before the specified time in `df_labels`

    Returns:
    --------
    list of lists
        For each row in df_labels, a list of codes that occur before its time for the same patient
    """
    # Convert Polars DataFrames to pandas
    df_data = df_data.to_pandas() if not isinstance(df_data, pd.DataFrame) else df_data
    df_labels = df_labels.to_pandas() if not isinstance(df_labels, pd.DataFrame) else df_labels

    # Sort both dataframes to ensure proper filtering
    df_data_sorted = df_data.sort_values(["subject_id", "time"])
    df_labels_sorted = df_labels.sort_values(["subject_id", "prediction_time"])

    # Create a list to store results
    prior_events: List[List[str]] = []

    # Group A dataframe by subject_id for efficient lookup
    grouped_a = df_data_sorted.groupby("subject_id")

    if n_procs == 1:
        prior_events = _process_prior_events_chunk(df_labels_sorted, grouped_a, is_show_tqdm=True)
    else:
        # Split df_labels_sorted into n chunks
        chunk_size = 1_000
        chunks = [df_labels_sorted.iloc[i : i + chunk_size] for i in range(0, len(df_labels_sorted), chunk_size)]

        # Create partial function with grouped_a already set
        process_chunk_partial = partial(_process_prior_events_chunk, grouped_a=grouped_a)

        # Process chunks in parallel
        print(f"Processing {len(chunks)} chunks across {n_procs} processes...")
        with multiprocessing.Pool(n_procs) as pool:
            results = list(tqdm(pool.imap(process_chunk_partial, chunks), total=len(chunks), desc="Processing chunks"))

        # Flatten results
        prior_events = [event for chunk_result in results for event in chunk_result]

    return prior_events


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a string using the GPT-4 tokenizer.
    """
    # Load the tokenizer for the specified model
    tokenizer = tiktoken.encoding_for_model(model)

    # Encode the text to get the tokens
    tokens = tokenizer.encode(text)

    # Return the number of tokens
    return len(tokens)


class EHRSHOTScenario(Scenario):
    """
    From "An EHR Benchmark for Few-Shot Evaluation of Foundation Models" (Wornow et al. 2023),
    EHRSHOT is a collection of structured data from 6,739 deidentified longitudinal
    electronic health records (EHRs) sourced from Stanford Medicine. It contains
    15 unique clinical prediction tasks. We use a subset of 14 of these tasks, namely
    the binary classification tasks.

    Citation
    ```
    @article{wornow2023ehrshot,
        title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models},
        author={Michael Wornow and Rahul Thapa and Ethan Steinberg and Jason Fries and Nigam Shah},
        year={2023},
        eprint={2307.02028},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    ```
    """

    name = "ehrshot"
    description = (
        "A dataset given a patient record of EHR codes, classifying if an event will occur at a future date or not."
    )
    tags = []  # TODO

    POSSIBLE_ANSWER_CHOICES: List[str] = [
        "yes",
        "no",
    ]

    def __init__(self, subject: str, max_length: Optional[int] = None):
        super().__init__()
        self.subject: str = subject  # same as "task" or "labeling_function"
        self.path_to_meds_dir: str = "/share/pi/nigam/data/medhelm/ehrshot/meds/"
        self.path_to_tmp_dir: str = "/share/pi/nigam/data/medhelm/ehrshot/prompts/"
        self.max_length = max_length

    def create_benchmark(self, n_procs: int = 4) -> Dict[str, str]:
        """Loads the MEDS dataset and converts it to prompts"""

        # Load MEDS EHRSHOT patient timelines
        df_data = pd.read_parquet(os.path.join(self.path_to_meds_dir, "data/data.parquet"))
        df_splits = pd.read_parquet(os.path.join(self.path_to_meds_dir, "metadata/subject_splits.parquet"))

        # Load MEDS EHRSHOT labels
        tasks = sorted(os.listdir(os.path.join(self.path_to_meds_dir, "labels")))
        for t in tasks:
            path_to_labels: str = os.path.join(self.path_to_meds_dir, "labels", t, "labels.parquet")
            if t != self.subject or not os.path.exists(path_to_labels):
                continue
            df_labels = pd.read_parquet(path_to_labels)

            # If lab value task, limit to 10k random labels b/c too many in EHRSHOT (upwards of 300k)
            if self.subject.startswith("lab_"):
                df_labels = df_labels.sample(n=CONFIG["max_labels_per_task"], random_state=CONFIG["seed"])

            # Create patient timelines, limited to only events prior to the prediction time of the label
            timelines_raw: List[List[str]] = get_prior_events(df_data, df_labels, n_procs=n_procs)
            timelines: List[List[Dict[str, Any]]] = [
                [{"code": code} for code in timeline] for timeline in timelines_raw
            ]
            assert (
                len(timelines) == df_labels.shape[0]
            ), f"Expected {df_labels.shape[0]} prior events, got {len(timelines)}"

        # Add splits
        df_labels["split"] = df_labels["subject_id"].map(df_splits["split"])

        # TODO -- Few-shot examples
        examples: List[Dict[str, Any]] = []
        n_shots = CONFIG.get("n_shots", 0)
        for i in range(n_shots if isinstance(n_shots, int) else 0):
            pass

        # Create LUMIA-ified prompt for each label
        print(f"Generating {len(timelines)} prompts...")
        prompts: List[str] = [lumia_prompt(self.subject, CONFIG, examples, {"ehr": x, "label": 0}) for x in timelines]
        df_labels["prompt"] = prompts

        # Save to parquet
        path_to_output_dir: str = os.path.join(self.path_to_tmp_dir, self.subject)
        ensure_directory_exists(path_to_output_dir)
        df_labels.to_parquet(os.path.join(path_to_output_dir, "medhelm_prompts.parquet"))
        return {"status": "success"}

    def get_instances(self, output_path: str) -> List[Instance]:
        path_to_input_csv: str = os.path.join(self.path_to_tmp_dir, self.subject, "medhelm_prompts.parquet")
        lock_path = path_to_input_csv + ".lock"
        with FileLock(lock_path):
            if not os.path.exists(path_to_input_csv):
                print(f"Creating benchmark from SCRATCH for {self.subject}...")
                self.create_benchmark()  # Create benchmark from scratch

        # Load data for this task
        df = pd.read_parquet(path_to_input_csv)

        # Generate instances
        instances: List[Instance] = []
        # df['prompt']=df['prompt'].str.replace('yes','A for yes').str.replace('no','B for no')
        for prompt, label, split in tqdm(
            zip(df["prompt"], df["boolean_value"], df["split"]), total=len(df), desc="Generating instances"
        ):
            if self.max_length is not None and count_tokens(prompt) > self.max_length:
                continue
            label = "yes" if label else "no"
            # split = TEST_SPLIT if split == "held_out" else (VALID_SPLIT if split == "tuning" else TRAIN_SPLIT)
            references: List[Reference] = [
                Reference(Output(text=pred_answer), tags=[CORRECT_TAG] if pred_answer == label else [])
                for pred_answer in EHRSHOTScenario.POSSIBLE_ANSWER_CHOICES
            ]
            instances.append(
                Instance(
                    input=Input(text=prompt),  # prompt
                    references=references,  # `plan` is the the label; `tags` is whether it is correct`
                    split=TEST_SPLIT,  # the split
                )
            )

        return instances


if __name__ == "__main__":
    # Generate statistics on prompts
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tqdm.pandas()
    n_procs: int = 10

    os.makedirs("./ehrshot_stats", exist_ok=True)
    for t in TASK_FULL_NAMES.keys():
        # Skip if already exists
        if os.path.exists(f"./ehrshot_stats/{t}.txt"):
            print(f"Skipping {t} because it already exists")
            continue

        # Create benchmark
        scenario = EHRSHOTScenario(subject=t)
        scenario.create_benchmark(n_procs=n_procs)
        instances = scenario.get_instances("test.csv")

        # Calculate prompt token stats
        path_to_input_csv = os.path.join(scenario.path_to_tmp_dir, scenario.subject, "medhelm_prompts.parquet")
        df = pd.read_parquet(path_to_input_csv)
        df["prompt_n_tokens"] = df["prompt"].progress_apply(lambda x: len(tokenizer.encode(x)))
        with open(f"./ehrshot_stats/{t}.txt", "w") as f:
            f.write("-" * 100 + "\n")
            f.write(f"Task: {t}\n")
            f.write(f"# of instances: {len(instances)}\n")
            f.write(f"# of positives: {df['boolean_value'].sum()}\n")
            f.write(f"Size of splits:\n{df['split'].value_counts()}\n")
            f.write(f"# tokens per prompt:\n{df['prompt_n_tokens'].describe()}\n")
            f.write("-" * 100 + "\n")
        df.to_parquet(os.path.join(scenario.path_to_tmp_dir, scenario.subject, "medhelm_prompts.parquet"))
