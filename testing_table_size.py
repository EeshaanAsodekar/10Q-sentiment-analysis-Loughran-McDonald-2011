# ----------------------------------------------------------
#  Sanity‑check: count Table‑I & Table‑II rows in all XMLs
# ----------------------------------------------------------
import os, glob, xml.etree.ElementTree as ET

XML_DIR = "form4_xml_2024_2025"          # change if needed

t1_rows = 0      # Table I  (non‑derivative)
t2_rows = 0      # Table II (derivative)

for fp in glob.glob(os.path.join(XML_DIR, "*.xml")):
    root = ET.parse(fp).getroot()

    # Table I rows: transactions + holdings
    t1_rows += len(root.findall("./nonDerivativeTable/nonDerivativeTransaction"))
    t1_rows += len(root.findall("./nonDerivativeTable/nonDerivativeHolding"))

    # Table II rows: transactions + holdings
    t2_rows += len(root.findall("./derivativeTable/derivativeTransaction"))
    t2_rows += len(root.findall("./derivativeTable/derivativeHolding"))

print(f"Table I rows (non‑derivative): {t1_rows:,}")
print(f"Table II rows (derivative):    {t2_rows:,}")
print(f"Grand total:                   {t1_rows + t2_rows:,}")
