# ----------------------------------------------------------
#  Sanity‑check: count ALL child elements under each table
# ----------------------------------------------------------
import os, glob, xml.etree.ElementTree as ET

XML_DIR = "form4_xml_2024_2025"

t1_count = 0   # all elements under nonDerivativeTable
t2_count = 0   # all elements under derivativeTable

for fp in glob.glob(os.path.join(XML_DIR, "*.xml")):
    root = ET.parse(fp).getroot()

    # count every direct child element of <nonDerivativeTable>
    t1_count += len(root.findall("./nonDerivativeTable/*"))

    # count every direct child element of <derivativeTable>
    t2_count += len(root.findall("./derivativeTable/*"))

print(f"Total elements under Table I (non‑derivative): {t1_count:,}")
print(f"Total elements under Table II (derivative):    {t2_count:,}")
print(f"Grand total elements:                           {t1_count + t2_count:,}")
