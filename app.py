from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import pickle
import os
import re
from werkzeug.utils import secure_filename
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'super_secret_key'
UPLOAD_FOLDER = 'uploads'
USER_FILE = 'users.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model + scaler + features
model = pickle.load(open("random_forest_best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_features = pd.read_csv("model_features.csv", header=None).squeeze().tolist()

# Input fields
selected_features = [
    'Animal_Type', 'Breed', 'Age', 'Gender', 'Weight',
    'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Duration',
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing',
    'Lameness', 'Skin_Lesions', 'Nasal_Discharge', 'Eye_Discharge',
    'Body_Temperature', 'Heart_Rate'
]


disease_medication_map = {
    "Actinobacillus Pleuropneumonia": "Targeted antibiotics (e.g., oxytetracycline/tulathromycin), NSAIDs, Supportive care",
    "Actinobacillus Suis": "Targeted antibiotics (e.g., oxytetracycline/tulathromycin), NSAIDs, Supportive care",
    "African Swine Fever": "No treatment, Biosecurity measures",
    "Allergic Rhinitis": "Supportive care; targeted therapy per diagnostics; consult veterinarian",
    "Arthritis": "NSAIDs, Disease-modifying agents, Weight control; joint injections for severe cases",
    "Blue Tongue": "Supportive care; vector control; anti-inflammatories",
    "Blue Tongue Disease": "Supportive care; vector control; anti-inflammatories",
    "Blue Tongue Virus": "Supportive care, NSAIDs, Fluids; antivirals if available; isolation",
    "Bluetongue": "Supportive care; vector control; anti-inflammatories",
    "Bluetongue Virus": "Supportive care, Insect control",
    "Bordetella Infection": "Antibiotics (e.g., doxycycline), Vaccination",
    "Bovine Coccidiosis": "Amprolium or sulfonamides; Fluids; environmental hygiene",
    "Bovine Influenza": "NSAIDs, Antibiotics for secondary infections",
    "Bovine Johne's Disease": "Supportive care; targeted therapy per diagnostics; consult veterinarian",
    "Bovine Leukemia Virus": "Supportive care, NSAIDs, Fluids; antivirals if available; isolation",
    "Bovine Mastitis": "Intramammary antibiotics, Anti-inflammatories",
    "Bovine Parainfluenza": "Supportive care, NSAIDs; antibiotics for secondary infections",
    "Bovine Pneumonia": "Broad-spectrum antibiotics, NSAIDs, Supportive care",
    "Bovine Respiratory Disease": "Broad-spectrum antibiotics, NSAIDs, Supportive care",
    "Bovine Respiratory Disease Complex": "Antibiotics, NSAIDs, Vaccination",
    "Bovine Respiratory Syncytial Virus": "Antipyretics, Bronchodilators, Supportive care",
    "Bovine Tuberculosis": "Culling/reportable; supportive care only",
    "Bovine Viral Diarrhea": "No specific treatment, Fluids, Preventive vaccination",
    "Canine Cough": "Antitussives, Antibiotics if bacterial, Rest",
    "Canine Distemper": "Supportive care, Anti-seizure meds, Fluids",
    "Canine Flu": "Supportive care, NSAIDs; antibiotics for secondary infections",
    "Canine Heartworm Disease": "Melarsomine, Doxycycline, Heartworm preventatives",
    "Canine Hepatitis": "Supportive care, Fluids, Liver support",
    "Canine Infectious Hepatitis": "Supportive care, Fluids, Hepatoprotectants",
    "Canine Influenza": "Supportive care, NSAIDs, Antibiotics for secondary infections",
    "Canine Leptospirosis": "Doxycycline or Penicillin",
    "Canine Parvovirus": "Aggressive fluid therapy, Antibiotics, Antiemetics",
    "Caprine Arthritis": "NSAIDs, Disease-modifying agents, Weight control",
    "Caprine Arthritis Encephalitis": "No cure; NSAIDs, Supportive care; management",
    "Caprine Arthritis Encephalitis Virus": "No cure; NSAIDs, Supportive care; management",
    "Caprine Pleuropneumonia": "Tylosin, Oxytetracycline",
    "Caprine Respiratory Disease": "Antibiotics (e.g., oxytetracycline), NSAIDs",
    "Caprine Viral Arthritis": "No cure; NSAIDs, Supportive care; management",
    "Caseous Lymphadenitis": "Drainage, Tulathromycin, Biosecurity",
    "Chlamydia in Sheep": "Tetracyclines (e.g., oxytetracycline), Supportive care",
    "Chronic Bronchitis": "Corticosteroids, Bronchodilators; manage irritants",
    "Coccidiosis": "Amprolium, Sulfa drugs (e.g., sulfadimethoxine)",
    "Conjunctivitis": "Topical antibiotics, Saline flushes",
    "Contagious Abortion": "Tetracycline (depending on cause), Quarantine",
    "Contagious Ecthyma": "Supportive care, Isolation, Wound care; vaccinate contacts",
    "Cryptosporidiosis": "Fluids/electrolytes; halofuginone where approved; hygiene control",
    "Dermatophilosis": "Keep skin dry, Topical antiseptics; antibiotics if secondary infection",
    "Distemper": "Supportive care, Anti-nausea meds, Isolation",
    "E. Coli Infection": "Fluids/electrolytes; antibiotics only if systemic; hygiene",
    "Echinococcosis": "Praziquantel/Albendazole (as directed); strict hygiene; zoonotic precautions",
    "Encephalitis": "Anti-inflammatories, Seizure control, Supportive care",
    "Enteritis": "Electrolytes, Antibiotics (if bacterial), Probiotics",
    "Equine Encephalitis": "Anti-inflammatories, Supportive care, Fluids",
    "Equine Encephalomyelitis": "Supportive care, Anti-inflammatories",
    "Equine Herpesvirus": "Antivirals (e.g., acyclovir), NSAIDs, Supportive care",
    "Equine Influenza Virus": "Rest, NSAIDs, Supportive care",
    "Equine Metabolic Syndrome": "Dietary management, Exercise",
    "Equine Pneumonia": "Antibiotics, Bronchodilators, NSAIDs",
    "Equine West Nile Virus": "Anti-inflammatories, Fluids, Supportive care",
    "Feline Asthma": "Corticosteroids, Bronchodilators",
    "Feline Calicivirus": "Supportive care, NSAIDs",
    "Feline Coronavirus": "Supportive care, Antivirals in some cases",
    "Feline Immunodeficiency Virus": "Supportive care, Treat secondary infections",
    "Feline Infectious Peritonitis": "GS-441524 (experimental), Supportive care",
    "Feline Leukemia": "Antiviral meds (e.g., AZT), Supportive care",
    "Feline Leukemia Virus": "Antivirals, Supportive care",
    "Feline Panleukopenia": "IV fluids, Antiemetics, Broad-spectrum antibiotics",
    "Feline Panleukopenia Virus": "IV fluids, Anti-nausea meds, Broad-spectrum antibiotics",
    "Feline Renal Disease": "Fluid therapy, Phosphate binders, Renal diet",
    "Feline Respiratory Disease Complex": "Supportive care; doxycycline if bacterial suspicion",
    "Feline Respiratory Infection": "Supportive care; doxycycline if bacterial suspicion",
    "Feline Rhinotracheitis": "Antivirals (famciclovir), Supportive care",
    "Feline Upper Respiratory Infection": "Antibiotics (e.g., doxycycline), Supportive care",
    "Feline Viral Rhinotracheitis": "Antivirals, Antibiotics, Lysine",
    "Foot and Mouth Disease": "No treatment, Strict control/quarantine",
    "Foot-and-Mouth Disease": "No treatment; quarantine/eradication measures",
    "Footrot": "Footbaths (zinc/copper sulfate), Trimming, Systemic antibiotics if severe",
    "Fungal Infection": "Itraconazole, Fluconazole",
    "Gastroenteritis": "Fluid therapy, Anti-nausea meds, Probiotics",
    "Gastrointestinal Infection": "Fluids/electrolytes; antibiotics only if bacterial",
    "Gastrointestinal Stasis": "Prokinetics (e.g. cisapride), Fluids, Pain relief",
    "Giardiasis": "Metronidazole, Fenbendazole",
    "Goat Pox": "No specific treatment, Supportive care",
    "Heartworm Disease": "Adulticide (melarsomine), Doxycycline, Preventatives",
    "Hyperthyroidism": "Methimazole, Radioactive iodine therapy",
    "Inflammatory Bowel Disease": "Prednisolone, Diet change, Metronidazole",
    "Intestinal Parasites": "Anthelmintics (e.g., fenbendazole, ivermectin)",
    "Johne's Disease": "No cure, Cull infected animals",
    "Kennel Cough": "Antitussives, Doxycycline if bacterial, Rest",
    "Laminitis": "NSAIDs, Cryotherapy (acute), Diet management, Stall rest",
    "Leptospirosis": "Doxycycline or Penicillin",
    "Lyme Disease": "Doxycycline",
    "Maedi-Visna": "No cure, Supportive care",
    "Mastitis": "Intramammary antibiotics, Anti-inflammatories",
    "Mannheimia Haemolytica Infection": "Oxytetracycline/tulathromycin; NSAIDs; Supportive care",
    "Metritis": "Broad-spectrum antibiotics, NSAIDs; supportive care",
    "Mycoplasma Infection": "Targeted antibiotics (e.g., tetracyclines/macrolides), NSAIDs",
    "Myxomatosis": "No treatment, Supportive care, Euthanasia often required",
    "Newcastle Disease": "No treatment; stamping out; supportive care in pets",
    "Ovine Chlamydiosis": "Tetracyclines; isolate; biosecurity",
    "Ovine Johne's Disease": "Supportive care; herd management; cull positives",
    "Ovine Parasitic Gastroenteritis": "Anthelmintics; pasture management; FAMACHA-guided deworming",
    "Pancreatitis": "Pain relief, Low-fat diet, IV fluids",
    "Parvovirus": "Aggressive IV fluids, Antiemetics, Antibiotics",
    "Pasteurellosis": "Penicillin, Oxytetracycline",
    "Peste des Petits Ruminants": "Supportive care; biosecurity; vaccination of contacts",
    "Pneumonia": "Antibiotics, Oxygen therapy, NSAIDs",
    "Porcine Circovirus Disease": "Vaccination, Supportive care",
    "Porcine Respiratory Disease Complex": "Antibiotics, Vaccination, Environmental management",
    "Q Fever": "Tetracyclines, Biosecurity; zoonotic precautions",
    "Rabbit Calicivirus": "Supportive care, Vaccination (where available)",
    "Rabbit Syphilis": "Penicillin G (avoid oral forms)",
    "Respiratory Syncytial Virus": "Antipyretics, Bronchodilators, Supportive care",
    "Ringworm": "Antifungals (e.g., itraconazole, miconazole)",
    "Rift Valley Fever": "Supportive care; vector control; anti-inflammatories",
    "Salmonellosis": "Electrolytes, Antibiotics (if needed)",
    "Scrapie": "No treatment; quarantine/eradication measures; supportive care where applicable",
    "Scrapie Disease": "No treatment; quarantine/eradication measures; supportive care where applicable",
    "Snuffles": "Broad-spectrum antibiotics if bacterial, NSAIDs, Nebulization/bronchodilators as needed",
    "Strangles": "Penicillin, Supportive care",
    "Swine Dysentery": "Fluids/electrolytes, Probiotics; antibiotics only if bacterial",
    "Swine Erysipelas": "Penicillin, Vaccination",
    "Swine Fever": "No treatment (highly contagious), Eradication strategy",
    "Swine Flu": "Supportive care, Antipyretics",
    "Swine Influenza": "Supportive care, NSAIDs, Fluids; antivirals if available; isolation",
    "Tick-Borne Disease": "Doxycycline, Tick prevention",
    "Tuberculosis": "Euthanasia (in most animals), Rare treatment options",
    "Upper Respiratory Infection": "Supportive care, Antibiotics if bacterial",
    "Viral Hemorrhagic Disease": "Supportive care, NSAIDs, Fluids; antivirals if available; isolation",
    "West Nile Virus": "Supportive care, Anti-inflammatories"
}



@app.route('/')
def home():
    return render_template('home.html')

# ========== LOGIN ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not os.path.exists(USER_FILE):
            error = "No users registered yet."
        else:
            df = pd.read_csv(USER_FILE)
            user = df[(df['username'] == username) & (df['password'] == password)]
            if not user.empty:
                session['user'] = username
                return redirect(url_for('dashboard'))
            else:
                error = "Invalid username or password"
    return render_template("login.html", error=error)

# ========== REGISTER ==========
@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            error = "Invalid email format"
        elif not username or not password:
            error = "All fields are required"
        else:
            if not os.path.exists(USER_FILE):
                df = pd.DataFrame(columns=['email', 'username', 'password'])
                df.to_csv(USER_FILE, index=False)

            users_df = pd.read_csv(USER_FILE)
            if username in users_df['username'].values:
                error = "Username already exists"
            else:
                new_user = pd.DataFrame([[email, username, password]], columns=['email', 'username', 'password'])
                new_user.to_csv(USER_FILE, mode='a', index=False, header=False)
                return redirect(url_for('login'))
    return render_template('register.html', error=error)

# ========== DASHBOARD ==========
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', features=selected_features)

# ========== PREDICT RESULT ==========
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Collect input data
    input_data = [request.form[str(i)] for i in range(len(selected_features))]
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Preprocess & scale
    encoded = preprocess(input_df)
    scaled = scaler.transform(encoded)

    # Get prediction probabilities
    probs = model.predict_proba(scaled)[0]
    top_indices = np.argsort(probs)[::-1][:3]  # Top 3 predictions
    top_diseases = [(model.classes_[i], round(probs[i] * 100, 2)) for i in top_indices]

    # Main predicted disease
    main_disease = top_diseases[0][0]
    medication = disease_medication_map.get(main_disease, "")

    return render_template("result.html",
                           result=main_disease,
                           medication=medication,
                           top_predictions=top_diseases)

# ---------- UPLOAD CSV ----------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    table_html = None
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)
        df = pd.read_csv(path)
        processed = preprocess(df)
        scaled = scaler.transform(processed)
        df['Predicted_Disease'] = model.predict(scaled)
        output_path = os.path.join(UPLOAD_FOLDER, 'predicted_output.csv')
        df.to_csv(output_path, index=False)
        session['csv_path'] = output_path
        table_html = df.to_html(classes='styled-table', index=False)
    return render_template('upload.html', table=table_html)

@app.route('/download')
def download():
    if 'csv_path' in session and os.path.exists(session['csv_path']):
        return send_file(session['csv_path'], as_attachment=True)
    return redirect(url_for('upload'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------- HELPERS ----------
def preprocess(df):
    """
    Ensures preprocessing exactly matches training.
    """
    cat_cols = ['Animal_Type', 'Breed', 'Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols)

    # Add missing columns from model training
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Remove any unexpected columns
    df = df[model_features]

    return df

if __name__ == '__main__':
    app.run(debug=False)