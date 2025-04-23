import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from fpdf import FPDF
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'
model = load_model("brain_tumor_model.h5")
class_labels = ['No Tumor', 'Tumor']

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "static/reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def generate_report(name, age, gender, contact, scan_type, scan_no, date, result_text):
    patient_id = str(uuid.uuid4())[:8].upper()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Atom Hospital", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "House 10, Road 2, Mirpur 1, Dhaka, Bangladesh", ln=True, align='C')
    pdf.cell(0, 10, "Phone: 01000000259", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "MRI/CT Scan Report", ln=True, align='C')
    pdf.ln(5)

    # Box for patient info
    pdf.set_font("Arial", size=12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(95, 10, f"Patient Name: {name}", 1, 0, 'L', 1)
    pdf.cell(95, 10, f"Age: {age}", 1, 1, 'L', 1)
    pdf.cell(95, 10, f"Gender: {gender}", 1, 0, 'L', 1)
    pdf.cell(95, 10, f"Contact: {contact}", 1, 1, 'L', 1)
    pdf.cell(95, 10, f"Scan Type: {scan_type}", 1, 0, 'L', 1)
    pdf.cell(95, 10, f"Date: {date}", 1, 1, 'L', 1)
    pdf.cell(95, 10, f"Scan No: {scan_no}", 1, 0, 'L', 1)
    pdf.cell(95, 10, f"Patient ID: {patient_id}", 1, 1, 'L', 1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Data:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "Complain of headache and dizziness.")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Protocol:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, "Image ware predicted by the brain tumar detecting model")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Findings:", ln=True)
    pdf.set_font("Arial", '', 12)
    if "No Tumor" in result_text:
        pdf.multi_cell(0, 10, "Everything seems normal")
    else:
        pdf.multi_cell(0, 10, "Located some unusual white part in the brain")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Conclusion:", ln=True)
    pdf.set_font("Arial", '', 12)
    if "No Tumor" in result_text:
        pdf.multi_cell(0, 10, "Tumor not detected")
    else:
        pdf.multi_cell(0, 10, "Tumor detected")

    pdf.ln(15)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Dr. X", ln=True, align='R')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "FCPS Consultant Radiologist", ln=True, align='R')

    filename = f"{patient_id}_{name.replace(' ', '_')}.pdf"
    path = os.path.join(REPORT_FOLDER, filename)
    pdf.output(path)
    return path, filename

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "lab" and password == "assistant":
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["image"]
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        contact = request.form.get("contact")
        scan_type = request.form.get("scan_type")

        if file and name and age and gender and contact and scan_type:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)
            class_idx = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            result = class_labels[class_idx]

            scan_no = str(uuid.uuid4())[:8].upper()
            patient_id = str(uuid.uuid4())[:8].upper()
            date = datetime.today().strftime("%Y-%m-%d")

            result_text = f"Prediction: {result}\nConfidence: {confidence:.2f}%"
            report_path, report_filename = generate_report(name, age, gender, contact, scan_type, scan_no, date, result_text)

            return render_template("result.html", name=name, age=age, gender=gender, contact=contact, patient_id=patient_id,
                                   scan_type=scan_type, date=date, scan_no=scan_no, result=result,
                                   confidence=confidence, report_filename=report_filename,
                                   image_path=filepath)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(REPORT_FOLDER, filename), as_attachment=True)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
