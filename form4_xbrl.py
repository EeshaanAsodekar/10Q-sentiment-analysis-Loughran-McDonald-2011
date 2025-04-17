# ==========================================================
#  Grab ALL 2024‑2025 Form 4 XMLs for the first 5 CIKs
# ----------------------------------------------------------
#  pip install pandas requests
# ==========================================================
import os, re, time, requests, pandas as pd
from urllib.parse import urljoin

CSV_FILE   = "sp500_CIKS.csv"              # must have column "CIK"
YEARS      = {2024, 2025}
DL_DIR     = "form4_xml_2024_2025"
USER_AGENT = "Eeshaan Asodekar (eshaan@example.com)"  # ASCII only
os.makedirs(DL_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept":     "application/xml",
    "Accept-Encoding": "gzip, deflate",
}

# ---------- polite request wrapper (<=5 req/s, retries) ---
LAST_HIT = 0.0
def throttle(max_rps=5):
    global LAST_HIT
    now = time.time()
    delay = 1.0 / max_rps - (now - LAST_HIT)
    if delay > 0:
        time.sleep(delay)
    LAST_HIT = time.time()

def http_get(url, want_json=False, tries=3):
    for _ in range(tries):
        throttle()
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            print("    429 Too Many Requests – sleeping 30 s")
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json() if want_json else r.content
    raise RuntimeError(f"{url} failed")

# ---------- find raw XML inside a filing directory --------
PREFS = ("doc4.xml", "primary_doc.xml", "form4.xml", "ownership.xml")
def choose_xml(base_url):
    """
    return relative xml path (root or one‑level subdir) or None
    """
    def items(url):
        return http_get(urljoin(url, "index.json"), want_json=True)["directory"]["item"]

    # search root
    files = [it["name"] for it in items(base_url)
             if it["type"] == "file" and it["name"].lower().endswith(".xml")]
    # search first‑level dirs (xslF345X##)
    for it in items(base_url):
        if it["type"] == "dir":
            sub_url = f"{base_url}/{it['name']}"
            for sub in items(sub_url):
                if sub["type"] == "file" and sub["name"].lower().endswith(".xml"):
                    files.append(f"{it['name']}/{sub['name']}")

    if not files:
        return None
    for p in PREFS:
        for f in files:
            if f.lower().endswith(p):
                return f
    return files[0]

# ---------- CIK list (first 5) ----------------------------
cik_df = pd.read_csv(CSV_FILE, dtype={"CIK": str}, encoding_errors="ignore")
cik_df["CIK"] = cik_df["CIK"].str.zfill(10)
first_five = cik_df["CIK"].head(5)

# ---------- main download loop ---------------------------
for cik in first_five:
    try:
        subm = http_get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        want_json=True)
        df = pd.DataFrame(subm["filings"]["recent"])
        df = df[df["reportDate"].str.strip() != ""]
        df["year"] = df["reportDate"].str[:4].astype(int)
        subset = df[(df["form"] == "4") & df["year"].isin(YEARS)]

        print(f"{cik}: {len(subset)} Form 4s in 2024‑25")

        for _, row in subset.iterrows():
            acc = row["accessionNumber"]
            acc_dir = acc.replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_dir}/"

            # if primaryDocument already xml & not in xsl folder
            xml_rel = row["primaryDocument"]
            if not xml_rel.lower().endswith(".xml") or "/xsl" in xml_rel:
                xml_rel = choose_xml(base)
            if not xml_rel:
                print(f"  {acc}: xml not found")
                continue

            raw_url = urljoin(base, xml_rel)
            # strip any xslF345X##/ to reach raw copy
            raw_url = re.sub(r"/xslF345[^/]+/", "/", raw_url)

            xml_bytes = http_get(raw_url)
            fname = f"{cik}_{row['reportDate']}_{os.path.basename(raw_url)}"
            with open(os.path.join(DL_DIR, fname), "wb") as f:
                f.write(xml_bytes)
            print(f"  saved {fname}")

    except Exception as e:
        print(f"{cik}: error {e}")
