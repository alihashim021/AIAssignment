
import re

MONEY_RE = re.compile(r'\$?\s*(\d+[\.,]\d{2})')
DATE_RE = re.compile(r'(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})')

def parse_receipt(text: str):
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in text.splitlines() if ln.strip()]
    merchant = lines[0] if lines else ''
    date_match = DATE_RE.search(text)
    monies = [m.group(1).replace(',', '.') for m in MONEY_RE.finditer(text)]
    total_val = max([float(m) for m in monies], default=None)
    return {
        'merchant': merchant,
        'date': date_match.group(1) if date_match else None,
        'total': round(total_val, 2) if total_val is not None else None,
    }
