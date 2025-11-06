#!/usr/bin/env python3
"""
Task 3 — W-2 Parser & Insight Generator (Gemini)

Module entrypoint:
    process_w2(file_path: str, test_mode: bool = False) -> dict

CLI:
    python w2_parser.py /path/to/W2.pdf
    python w2_parser.py /path/to/W2.jpg --test
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Optional OCR deps (safe import with fallbacks) ----------
try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:
    convert_from_bytes = None  # graceful fallback

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:
    pytesseract = None
    Image = None

# ---------- Gemini client (safe import) ----------
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None
# ---------- Constants & helpers ----------
NUMERIC_BOXES = {
    # federal boxes 1-14 (not all are always present)
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12a", "12b", "12c", "12d", "13",
    "14",
    # state & local
    "16", "17", "18", "19", "20"
}

# Social Security wage base caps (by year)
SS_WAGE_BASE = {
    2021: 142800,
    2022: 147000,
    2023: 160200,
    2024: 168600,
    # Keep 2025+ empty to avoid stale assertions; year not found => skip cap check
}

STATE_CODES = {
    # 2-letter USPS codes (50 states + DC)
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
}

ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")

MASK_SSN = re.compile(r"\b(\d{3})[- ]?(\d{2})[- ]?(\d{4})\b")
MASK_EIN = re.compile(r"\b(\d{2})[- ]?(\d{7})\b")

DEFAULT_MODEL = "gemini-2.5-flash"

@dataclass
class QualityNotes:
    warnings: List[str]
    ocr_confidence: Optional[float] = None
    sent_to_gemini: bool = False


# ---------- Privacy helpers ----------
def mask_last4_ssn(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) == 9:
        return "*-**-" + digits[-4:]
    if len(digits) >= 4:
        return "*-**-" + digits[-4:]
    return None

def mask_last4_ein(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return "*-**-" + digits[-4:]
    return None

def mask_all_ids_in_text(text: str) -> str:
    text = MASK_SSN.sub(lambda m: "*-**-" + m.group(3), text)
    text = MASK_EIN.sub(lambda m: "*-**-" + m.group(2)[-4:], text)
    return text


# ---------- OCR ----------
def ocr_pdf_or_image(file_bytes: bytes, mime_type: str, quality: QualityNotes) -> Optional[str]:
    """
    Best-effort OCR:
      - If PDF: convert pages to images then OCR
      - If image: OCR directly
    Fails gracefully if deps are missing.
    """
    if pytesseract is None or Image is None:
        quality.warnings.append("OCR not available (pytesseract/Pillow not installed).")
        return None

    try:
        if mime_type == "application/pdf":
            if convert_from_bytes is None:
                quality.warnings.append("PDF-to-image not available (pdf2image not installed).")
                return None
            pages = convert_from_bytes(file_bytes, dpi=300)
            texts = []
            for img in pages:
                texts.append(pytesseract.image_to_string(img))
            text = "\n".join(texts)
        else:
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
        # pytesseract doesn't give confidence in simple API; leave None
        return text
    except Exception as e:
        quality.warnings.append(f"OCR error: {str(e)[:120]}")
        return None


# ---------- Normalization ----------
def _to_float(v: Any) -> Optional[float]:
    if v in (None, "", "null"):
        return None
    try:
        s = str(v).replace(",", "").strip()
        return float(s)
    except Exception:
        return None

def _norm_state_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip().upper()
    return c if c in STATE_CODES else c  # keep original even if non-standard; warning later

def _coerce_fields(struct: Dict[str, Any], quality: QualityNotes) -> Dict[str, Any]:
    """Coerce numbers, title-case names, normalize state codes, and collect warnings."""
    out = json.loads(json.dumps(struct))  # deep copy

    # Employee
    emp = out.get("employee", {}) or {}
    emp["name"] = (emp.get("name") or "").strip()
    if emp.get("address"):
        addr = emp["address"]
        addr["state"] = _norm_state_code(addr.get("state"))
        # basic zip normalization
        if addr.get("zip"):
            addr["zip"] = str(addr["zip"]).strip()
        out["employee"]["address"] = addr
    ssn_masked = mask_last4_ssn(emp.get("ssn"))
    out["employee"]["ssn"] = ssn_masked

    # Employer
    er = out.get("employer", {}) or {}
    er["name"] = (er.get("name") or "").strip()
    er["ein"] = mask_last4_ein(er.get("ein"))
    out["employer"] = er

    # Federal boxes
    fed = out.get("federal", {}) or {}
    # normalize boxed numeric values
    for key, val in list(fed.items()):
        # box 12 can be object; box 13 flags; 14 can be str or map
        if key in {"boxes_12", "other", "box_13"}:
            continue
        fed[key] = _to_float(val)
        if val is not None and fed[key] is None:
            quality.warnings.append(f"Non-numeric in federal box {key}: {val}")

    out["federal"] = fed

    # States (list)
    states = out.get("state") or []
    norm_states = []
    for s in states:
        s = s or {}
        s["state_code"] = _norm_state_code(s.get("state_code"))
        s["employer_state_id"] = (s.get("employer_state_id") or "").strip() or None
        s["state_wages"] = _to_float(s.get("state_wages"))
        s["state_income_tax"] = _to_float(s.get("state_income_tax"))
        norm_states.append(s)
    out["state"] = norm_states

    # Locals (list)
    locals_ = out.get("local") or []
    norm_locals = []
    for l in locals_:
        l = l or {}
        l["locality_name"] = (l.get("locality_name") or "").strip() or None
        l["local_wages"] = _to_float(l.get("local_wages"))
        l["local_income_tax"] = _to_float(l.get("local_income_tax"))
        norm_locals.append(l)
    out["local"] = norm_locals

    # Missing boxes warnings (federal 1-6 are common)
    for must in ["wages_tips_other_comp", "federal_income_tax_withheld",
                 "social_security_wages", "social_security_tax_withheld",
                 "medicare_wages_and_tips", "medicare_tax_withheld"]:
        if out.get("federal", {}).get(must) is None:
            quality.warnings.append(f"Missing federal box: {must}")

    # State code sanity
    for s in out["state"]:
        sc = s.get("state_code")
        if sc and sc not in STATE_CODES:
            quality.warnings.append(f"Non-standard state code: {sc}")

    return out


# ---------- Gemini calls ----------
def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        return genai.Client(api_key="AIzaSyA_QydWs-UOBDE4FuLMJDywcb4ndtUrLTQ")
    except Exception:
        return None

def _retry_call(fn, retries=1, delay=1.0):
    last = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if i < retries:
                time.sleep(delay)
    raise last

def _load_prompt_text(rel_path: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, rel_path)
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def _clean_json_text(raw_text: str) -> str:
    # Sometimes models wrap with ```json ...```
    s = raw_text.strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"```$", "", s)
    s = re.sub(r"^json\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def _extract_with_gemini(file_bytes: bytes, mime_type: str, ocr_text: Optional[str], quality: QualityNotes) -> Dict[str, Any]:
    client = _get_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable or GEMINI_API_KEY not set.")

    prompt = _load_prompt_text("prompts/extract_w2.txt")

    def do_call():
        parts = []
        # Primary: attach file (PDF/image)
        if genai_types is not None:
            parts.append(genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        # Secondary: include OCR text if we have it (gives model extra signal)
        if ocr_text:
            parts.append(f"OCR_TEXT:\n{mask_all_ids_in_text(ocr_text)[:8000]}")  # keep payload reasonable
        parts.append(prompt)

        resp = client.models.generate_content(model=DEFAULT_MODEL, contents=parts)
        return resp

    resp = _retry_call(do_call, retries=1, delay=1.2)
    quality.sent_to_gemini = True

    text = _clean_json_text(resp.text or "")
    try:
        data = json.loads(text)
    except Exception:
        # If the model failed strict JSON, try to salvage with simple braces extraction
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Gemini extraction did not return valid JSON.")
        data = json.loads(m.group(0))

    # Ensure masking for safety
    fld = data.get("fields", {})
    if "employee" in fld:
        fld["employee"]["ssn"] = mask_last4_ssn(fld["employee"].get("ssn"))
    if "employer" in fld:
        fld["employer"]["ein"] = mask_last4_ein(fld["employer"].get("ein"))
    data["fields"] = fld
    return data

def _insights_with_gemini(fields_json: Dict[str, Any], quality: QualityNotes) -> List[str]:
    client = _get_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable or GEMINI_API_KEY not set.")

    prompt = _load_prompt_text("prompts/insights_w2.txt")
    payload = json.dumps(fields_json, ensure_ascii=False)

    def do_call():
        resp = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[payload, prompt],
        )
        return resp

    resp = _retry_call(do_call, retries=1, delay=1.2)
    txt = _clean_json_text(resp.text or "")

    try:
        out = json.loads(txt)
        if isinstance(out, list):
            return [str(x) for x in out]
        # If it sent {"insights":[...]}
        if isinstance(out, dict) and isinstance(out.get("insights"), list):
            return [str(x) for x in out["insights"]]
        return [txt.strip()]
    except Exception:
        return [txt.strip()[:1000]]


# ---------- Local (offline) test data ----------
def _dummy_fields() -> Dict[str, Any]:
    return {
        "fields": {
            "employee": {
                "name": "John Doe",
                "ssn": "*-**-1234",
                "address": {"street": "123 Main St", "city": "Anytown", "state": "CA", "zip": "90001"}
            },
            "employer": {
                "name": "DesignNext",
                "ein": "*-**-8788",
                "address": "Katham Dorbosto, Kashiani, Gopalgonj, AK 8133"
            },
            "federal": {
                "wages_tips_other_comp": 80000.00,
                "federal_income_tax_withheld": 10368.00,
                "social_security_wages": 80000.00,
                "social_security_tax_withheld": 4960.00,
                "medicare_wages_and_tips": 80000.00,
                "medicare_tax_withheld": 1160.00,
                "social_security_tips": None,
                "allocated_tips": None,
                "advanced_eic_payments": None,
                "dependent_care_benefits": None,
                "nonqualified_plans": None,
                "boxes_12": {
                    "a": {"code": None, "amount": None},
                    "b": {"code": None, "amount": None},
                    "c": {"code": None, "amount": None},
                    "d": {"code": None, "amount": None},
                },
                "other": {"14": None}
            },
            "state": [{
                "state_code": "AL",
                "employer_state_id": "877878878",
                "state_wages": 80000.00,
                "state_income_tax": 835.00
            }],
            "local": []
        }
    }

def _dummy_insights() -> List[str]:
    return [
        "ZIP format looks valid: 90001 (CA).",
        "Federal withholding ~12.96% of Box 1 (within common ranges 8–22%).",
        "Social Security wages ($80,000) below 2022 cap ($147,000); no cap alert.",
        "No multi-state income detected.",
        "Medicare wages equal Social Security wages; consistent.",
    ]


# ---------- Post-parse derived checks ----------
def _derived_insights_locally(fields: Dict[str, Any], quality: QualityNotes) -> None:
    """Add warnings and consistency checks we can compute locally (independent of Gemini)."""
    fed = fields.get("federal", {}) or {}
    b1 = _to_float(fed.get("wages_tips_other_comp"))
    b2 = _to_float(fed.get("federal_income_tax_withheld"))
    b3 = _to_float(fed.get("social_security_wages"))
    b5 = _to_float(fed.get("medicare_wages_and_tips"))

    # Withholding heuristic alert if wildly outside typical 5%–30%
    if b1 and b2 is not None:
        pct = (b2 / b1) * 100 if b1 else None
        if pct is not None and (pct < 5 or pct > 30):
            quality.warnings.append(f"Federal withholding unusual ratio: {pct:.2f}% of Box 1.")

    # SS cap check
    # Try to identify tax year from any visible "year" fields; else attempt heuristic: presence of '2022' etc. in employer address or other
    year = None
    for k in ("tax_year", "year"):
        if isinstance(fed.get(k), (int, float)):
            year = int(fed[k])
            break
    if year is None:
        # heuristic: scan other text containers for 20xx
        text = json.dumps(fields)  # safe; masked already
        m = re.search(r"\b(20[12]\d)\b", text)
        if m:
            try:
                year = int(m.group(1))
            except Exception:
                year = None

    if year in SS_WAGE_BASE and b3 is not None:
        cap = SS_WAGE_BASE[year]
        if b3 >= max(0.95 * cap, cap - 1000):  # near cap heuristic
            quality.warnings.append(f"Box 3 Social Security wages near {year} wage base cap (${cap:,}).")

    # Address sanity checks
    addr = (fields.get("employee", {}) or {}).get("address") or {}
    state = (addr.get("state") or "").upper()
    zipc = addr.get("zip")
    if state and state not in STATE_CODES:
        quality.warnings.append(f"Employee address: non-standard state code '{state}'.")
    if zipc and not ZIP_RE.match(str(zipc)):
        quality.warnings.append(f"Employee address: ZIP looks non-US format '{zipc}'.")


# ---------- Public entrypoint ----------
def process_w2(file_path: str, test_mode: bool = False) -> Dict[str, Any]:
    # Detect mime
    mime_type = "application/pdf" if file_path.lower().endswith(".pdf") else "image/jpeg"

    # Read bytes without persisting
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    quality = QualityNotes(warnings=[], ocr_confidence=None, sent_to_gemini=not test_mode)

    # OCR (best-effort, optional)
    ocr_text = ocr_pdf_or_image(file_bytes, mime_type, quality)

    if test_mode:
        fields = _dummy_fields()
        insights = _dummy_insights()
        result_fields = _coerce_fields(fields.get("fields", {}), quality)
        _derived_insights_locally(result_fields, quality)
        return {"fields": result_fields, "insights": insights, "quality": quality.__dict__}

    # Online path (Gemini)
    try:
        ext = _extract_with_gemini(file_bytes, mime_type, ocr_text, quality)
    except Exception as e:
        # Fail gracefully but safely
        msg = mask_all_ids_in_text(str(e))
        return {"fields": {}, "insights": [], "quality": {"warnings": [f"Gemini extraction error: {msg}"], "sent_to_gemini": False}}

    fields = ext.get("fields", {})
    fields = _coerce_fields(fields, quality)
    _derived_insights_locally(fields, quality)

    try:
        insights = _insights_with_gemini(fields, quality)
    except Exception as e:
        insights = []
        quality.warnings.append(f"Gemini insights error: {mask_all_ids_in_text(str(e))[:160]}")

    return {"fields": fields, "insights": insights, "quality": quality.__dict__}


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="W-2 Parser & Insight Generator (Gemini)")
    parser.add_argument("file", help="Path to W-2 PDF or image (e.g., /mnt/data/taskpng.jpg)")
    parser.add_argument("--test", action="store_true", help="Run offline test mode (no network calls)")
    args = parser.parse_args()

    res = process_w2(args.file, test_mode=args.test)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
