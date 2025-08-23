import pdfplumber
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Convert PDF tables to CSV.")
parser.add_argument("pdf_path", help="Path to the input PDF file")
parser.add_argument("csv_path", help="Path to the output CSV file")
args = parser.parse_args()

pdf_path = args.pdf_path
csv_path = args.csv_path

data = []

def filter_empty_rows(data):
    return [row for row in data if any(cell is not None and str(cell).strip() != "" for cell in row)]

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                data.append(row)

# Remove empty rows
data = filter_empty_rows(data)

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False, header=False)

print(f"CSV saved to {csv_path}")