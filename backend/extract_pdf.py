from pypdf import PdfReader

pdf_path = "data/pmc_diabetes.txt.pdf"
output_path = "data/pmc_diabetes.txt"

reader = PdfReader(pdf_path)

with open(output_path, "w", encoding="utf-8") as f:
    for page in reader.pages:
        text = page.extract_text()
        if text:
            f.write(text + "\n")

print("PDF text extracted successfully.")
