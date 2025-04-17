# ==========================================================
#  Parse Form 4 ownership XML  →  DataFrame
#  ---------------------------------------------------------
#  * Columns common to every row
#      cik               (issuer / company)
#      accession         (taken from the file name)
#      period_of_report  (YYYY‑MM‑DD)
#      owner_cik
#      owner_name
#      owner_relation    (comma‑separated flags + officer title)
#      table             ("NonDerivative" / "Derivative")
#
#  * Plus every element that can appear in a transaction:
#      security_title, transaction_date, transaction_code,
#      transaction_shares, transaction_price, acquired_disposed,
#      shares_owned_following, direct_or_indirect,
#      underlying_title, underlying_shares
#
#  You can add/remove fields in FIELDS_NONDERIV / FIELDS_DERIV
#  ----------------------------------------------------------
#  pip install pandas
# ==========================================================
import os, glob, re, xml.etree.ElementTree as ET
import pandas as pd

XML_DIR = "form4_xml_2024_2025"   # where the *.xml files live
OUTFILE = "form4_transactions.parquet"

# -------- field maps --------------------------------------
FIELDS_NONDERIV = dict(
    security_title         = "./securityTitle/value",
    transaction_date       = "./transactionDate/value",
    transaction_code       = "./transactionCoding/transactionCode",
    transaction_shares     = "./transactionAmounts/transactionShares/value",
    transaction_price      = "./transactionAmounts/transactionPricePerShare/value",
    acquired_disposed      = "./transactionAmounts/transactionAcquiredDisposedCode/value",
    shares_owned_following = "./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect     = "./ownershipNature/directOrIndirectOwnership/value",
)

FIELDS_DERIV = dict(
    security_title         = "./securityTitle/value",
    transaction_date       = "./transactionDate/value",
    transaction_code       = "./transactionCoding/transactionCode",
    transaction_shares     = "./transactionAmounts/transactionShares/value",
    transaction_price      = "./transactionAmounts/transactionPricePerShare/value",
    acquired_disposed      = "./transactionAmounts/transactionAcquiredDisposedCode/value",
    shares_owned_following = "./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect     = "./ownershipNature/directOrIndirectOwnership/value",
    underlying_title       = "./underlyingSecurity/underlyingSecurityTitle/value",
    underlying_shares      = "./underlyingSecurity/underlyingSecurityShares/value",
)

# -------- helpers ----------------------------------------
def text_at(elem, path):
    tgt = elem.find(path)
    return (tgt.text or "").strip() if tgt is not None else None

def owner_relation(owner):
    rel = owner.find("./reportingOwnerRelationship")
    pieces = []
    if text_at(rel, "isDirector") == "1":        pieces.append("Director")
    if text_at(rel, "isOfficer") == "1":         pieces.append("Officer")
    if text_at(rel, "isTenPercentOwner") == "1": pieces.append("10% Owner")
    if text_at(rel, "isOther") == "1":           pieces.append("Other")
    off_title = text_at(rel, "officerTitle")
    if off_title: pieces.append(off_title)
    return ", ".join(pieces)

def parse_one(xml_path):
    rows = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cik     = text_at(root, "./issuer/issuerCik")
    period  = text_at(root, "./periodOfReport")

    # guess accession from file name  e.g. CIK_YYYY-MM-DD_doc4.xml → middle part
    base      = os.path.basename(xml_path)
    parts     = base.split("_", 2)
    accession = parts[2] if len(parts) > 2 else base

    # could be multiple owners (rare but valid)
    for owner in root.findall("./reportingOwner"):
        ocik  = text_at(owner, "./reportingOwnerId/rptOwnerCik")
        oname = text_at(owner, "./reportingOwnerId/rptOwnerName")
        rel   = owner_relation(owner)

        # ---------- Non‑Derivative (Table I) ----------
        for tx in root.findall("./nonDerivativeTable/nonDerivativeTransaction"):
            rec = {
                "cik": cik,
                "accession": accession,
                "period_of_report": period,
                "owner_cik": ocik,
                "owner_name": oname,
                "owner_relation": rel,
                "table": "NonDerivative",
            }
            for col, xpath in FIELDS_NONDERIV.items():
                rec[col] = text_at(tx, xpath)
            rows.append(rec)

        # ---------- Derivative (Table II) -------------
        for tx in root.findall("./derivativeTable/derivativeTransaction"):
            rec = {
                "cik": cik,
                "accession": accession,
                "period_of_report": period,
                "owner_cik": ocik,
                "owner_name": oname,
                "owner_relation": rel,
                "table": "Derivative",
            }
            for col, xpath in FIELDS_DERIV.items():
                rec[col] = text_at(tx, xpath)
            rows.append(rec)
    return rows

# -------- iterate every downloaded XML -------------------
all_rows = []
for xml_file in glob.glob(os.path.join(XML_DIR, "*.xml")):
    try:
        all_rows.extend(parse_one(xml_file))
    except Exception as e:
        print(f"parse error {xml_file}: {e}")

df = pd.DataFrame(all_rows)
df.to_csv("test_form4.csv",index=False)
df.to_parquet(OUTFILE, index=False)   # feather/CSV also fine
print(f"{len(df)} transaction rows written to {OUTFILE}")
