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
import os, glob, xml.etree.ElementTree as ET, pandas as pd

XML_DIR = "form4_xml_2024_2025"
OUTFILE = "form4_transactions.parquet"

# ----- field templates ------------------------------------
FIELDS_TX = dict(   # common to TX rows
    security_title="./securityTitle/value",
    transaction_date="./transactionDate/value",
    transaction_code="./transactionCoding/transactionCode",
    transaction_shares="./transactionAmounts/transactionShares/value",
    transaction_price="./transactionAmounts/transactionPricePerShare/value",
    acquired_disposed="./transactionAmounts/transactionAcquiredDisposedCode/value",
    shares_owned_following="./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect="./ownershipNature/directOrIndirectOwnership/value",
)
FIELDS_HOLD = dict(  # holdings (no txn info)
    security_title="./securityTitle/value",
    shares_owned_following="./postTransactionAmounts/sharesOwnedFollowingTransaction/value",
    direct_or_indirect="./ownershipNature/directOrIndirectOwnership/value",
    nature_of_ownership="./ownershipNature/natureOfOwnership/value",
)
# extra for derivative rows
UNDERLYING = dict(
    underlying_title="./underlyingSecurity/underlyingSecurityTitle/value",
    underlying_shares="./underlyingSecurity/underlyingSecurityShares/value",
)

# ----- helpers --------------------------------------------
def text_at(elem, path):
    tgt = elem.find(path)
    return (tgt.text or "").strip() if tgt is not None and tgt.text else None
def truthy(v): return str(v).lower().strip() in {"1","true","yes"}
def owner_relation(o):
    rel=o.find("./reportingOwnerRelationship"); parts=[]
    if rel is None: return None
    if truthy(text_at(rel,"isDirector")): parts.append("Director")
    if truthy(text_at(rel,"isOfficer")):  parts.append("Officer")
    if truthy(text_at(rel,"isTenPercentOwner")): parts.append("10% Owner")
    if truthy(text_at(rel,"isOther")): parts.append("Other")
    for tag in ("officerTitle","otherText"):
        if (t:=text_at(rel,tag)): parts.append(t)
    return ", ".join(parts) if parts else None

# ----- parser ---------------------------------------------
def parse_file(path):
    r=ET.parse(path).getroot()
    base=os.path.basename(path)
    accession=base.split("_",2)[-1]
    meta_core=dict(
        cik=text_at(r,"./issuer/issuerCik"),
        period_of_report=text_at(r,"./periodOfReport"),
        accession=accession,
    )
    rows=[]
    for own in r.findall("./reportingOwner"):
        meta=dict(**meta_core,
            owner_cik=text_at(own,"./reportingOwnerId/rptOwnerCik"),
            owner_name=text_at(own,"./reportingOwnerId/rptOwnerName"),
            owner_relation=owner_relation(own),
        )
        # Table I transactions
        for tx in r.findall("./nonDerivativeTable/nonDerivativeTransaction"):
            rec={**meta,"table":"NonDerivative"}
            for k,x in FIELDS_TX.items(): rec[k]=text_at(tx,x)
            rows.append(rec)
        # Table I holdings
        for hd in r.findall("./nonDerivativeTable/nonDerivativeHolding"):
            rec={**meta,"table":"NonDerivative"}
            for k,x in FIELDS_HOLD.items(): rec[k]=text_at(hd,x)
            rows.append(rec)
        # Table II transactions
        for tx in r.findall("./derivativeTable/derivativeTransaction"):
            rec={**meta,"table":"Derivative"}
            for k,x in {**FIELDS_TX,**UNDERLYING}.items():
                rec[k]=text_at(tx,x)
            rows.append(rec)
        # Table II holdings
        for hd in r.findall("./derivativeTable/derivativeHolding"):
            rec={**meta,"table":"Derivative"}
            for k,x in {**FIELDS_HOLD,**UNDERLYING}.items():
                rec[k]=text_at(hd,x)
            rows.append(rec)
    return rows

# ----- run -------------------------------------------------
rows=[]
for fp in glob.glob(os.path.join(XML_DIR,"*.xml")):
    try: rows.extend(parse_file(fp))
    except Exception as e: print("error",fp,e)

pd.DataFrame(rows).to_csv("test_form4.csv",index=False)
print(pd.DataFrame(rows).shape)
