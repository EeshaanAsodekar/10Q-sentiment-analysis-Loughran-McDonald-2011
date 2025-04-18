# ==========================================================
#  Form 4 XML  →  two DataFrames
#    • transactions_df  (all *Transaction* rows)
#    • holdings_df      (all *Holding*    rows)
# ----------------------------------------------------------
#  pip install pandas
# ==========================================================
import os, glob, xml.etree.ElementTree as ET, pandas as pd

XML_DIR   = "form4_xml_2024_2025"           # <‑‑ your folder
TX_FILE   = "transactions.csv"
HOLD_FILE = "holdings.csv"

# ---------- field maps ------------------------------------
TX_FIELDS = dict(   # for *_Transaction
    security_title="./securityTitle/value",
    transaction_date="./transactionDate/value",
    transaction_code="./transactionCoding/transactionCode",
    transaction_shares="./transactionAmounts/transactionShares/value",
    transaction_price="./transactionAmounts/transactionPricePerShare/value",
    acquired_disposed="./transactionAmounts/transactionAcquiredDisposedCode/value",
    shares_owned_following="./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect="./ownershipNature/directOrIndirectOwnership/value",
    # UNDERLYING gets added only for derivative rows
)
HOLD_FIELDS = dict( # for *_Holding
    security_title="./securityTitle/value",
    shares_owned_following="./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect="./ownershipNature/directOrIndirectOwnership/value",
    nature_of_ownership="./ownershipNature/natureOfOwnership/value",
)

UNDERLYING = dict(
    underlying_title ="./underlyingSecurity/underlyingSecurityTitle/value",
    underlying_shares="./underlyingSecurity/underlyingSecurityShares/value",
)

# ---------- helpers ---------------------------------------
def text_at(elem, path):
    tgt=elem.find(path)
    return (tgt.text or "").strip() if tgt is not None and tgt.text else None

def truthy(v): return str(v).strip().lower() in {"1","true","yes"}

def owner_relation(owner):
    rel=owner.find("./reportingOwnerRelationship")
    if rel is None: return None
    parts=[]
    if truthy(text_at(rel,"isDirector")):        parts.append("Director")
    if truthy(text_at(rel,"isOfficer")):         parts.append("Officer")
    if truthy(text_at(rel,"isTenPercentOwner")): parts.append("10% Owner")
    if truthy(text_at(rel,"isOther")):           parts.append("Other")
    for tag in ("officerTitle","otherText"):
        if (t:=text_at(rel,tag)): parts.append(t)
    return ", ".join(parts) if parts else None

# ---------- parser ----------------------------------------
def parse_xml(file_path):
    root = ET.parse(file_path).getroot()
    base = os.path.basename(file_path)
    accession = base.split("_", 2)[-1]

    meta_core = dict(
        cik             = text_at(root, "./issuer/issuerCik"),
        period_of_report= text_at(root, "./periodOfReport"),
        accession       = accession,
    )

    tx_rows   = []
    hold_rows = []

    for owner in root.findall("./reportingOwner"):
        meta = dict(**meta_core,
            owner_cik     = text_at(owner, "./reportingOwnerId/rptOwnerCik"),
            owner_name    = text_at(owner, "./reportingOwnerId/rptOwnerName"),
            owner_relation= owner_relation(owner),
        )

        # ---- Table I
        for tx in root.findall("./nonDerivativeTable/nonDerivativeTransaction"):
            rec={**meta, "table":"NonDerivative"}
            for k,x in TX_FIELDS.items(): rec[k]=text_at(tx,x)
            tx_rows.append(rec)

        for hd in root.findall("./nonDerivativeTable/nonDerivativeHolding"):
            rec={**meta, "table":"NonDerivative"}
            for k,x in HOLD_FIELDS.items(): rec[k]=text_at(hd,x)
            hold_rows.append(rec)

        # ---- Table II
        for tx in root.findall("./derivativeTable/derivativeTransaction"):
            rec={**meta, "table":"Derivative"}
            for k,x in {**TX_FIELDS,**UNDERLYING}.items():
                rec[k]=text_at(tx,x)
            tx_rows.append(rec)

        for hd in root.findall("./derivativeTable/derivativeHolding"):
            rec={**meta, "table":"Derivative"}
            for k,x in {**HOLD_FIELDS,**UNDERLYING}.items():
                rec[k]=text_at(hd,x)
            hold_rows.append(rec)

    return tx_rows, hold_rows

# ---------- run over all XMLs ------------------------------
transactions = []
holdings     = []

for fp in glob.glob(os.path.join(XML_DIR, "*.xml")):
    try:
        tx, hd = parse_xml(fp)
        transactions.extend(tx)
        holdings.extend(hd)
    except Exception as e:
        print("parse error", fp, e)

transactions_df = pd.DataFrame(transactions)
holdings_df     = pd.DataFrame(holdings)

transactions_df.to_csv(TX_FILE, index=False)
holdings_df.to_csv(HOLD_FILE, index=False)

print(f"saved {len(transactions_df):,} transaction rows  → {TX_FILE}")
print(f"saved {len(holdings_df):,}   holding rows       → {HOLD_FILE}")
print(f"total:{len(transactions_df) + len(holdings_df)}")
