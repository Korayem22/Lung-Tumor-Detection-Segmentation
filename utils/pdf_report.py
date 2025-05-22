import os
from fpdf import FPDF

def generate_report(output_path, existence, bboxes, area, fragments):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Tumor Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Tumor Present: {existence}", ln=True)
    pdf.cell(200, 10, txt=f"Bounding Boxes: {bboxes}", ln=True)
    pdf.cell(200, 10, txt=f"Total Tumor Area: {area} px", ln=True)
    pdf.cell(200, 10, txt=f"Tumor Fragments: {fragments}", ln=True)

    pdf.output(output_path)
