import fitz 
import pdfplumber

text_pages = []

with pdfplumber.open(r"Data\AllArticles_31Dec2025_compressed.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_pages.append(text)

full_text = "\n".join(text_pages)

with open("nyc_zoning_raw.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text()
    
    return text

text = extract_text_from_pdf(r"Data\AllArticles_31Dec2025_compressed.pdf")