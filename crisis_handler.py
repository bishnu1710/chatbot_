# crisis_handler.py
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)

HELPLINES = {
    "US": {"country":"United States","emergency":"911","suicide_crisis":"988 (call or text)","other":[]},
    "IN": {"country":"India","emergency":"112","suicide_crisis":"9152987821","other":[]},
    "DEFAULT": {"country":"International","emergency":"112 or 911","suicide_crisis":"Contact local emergency services","other":[]}
}

CRISIS_KEYWORDS = {
    "suicide": [r"\bkill myself\b", r"\bi want to die\b", r"\bcommit suicide\b", r"\bi'll kill myself\b", r"\bi will kill myself\b", r"\bkill me\b"],
    "self_harm": [r"\bcut myself\b", r"\bself[- ]harm\b", r"\bcutting\b"],
    "danger": [r"\bhelp me\b.*\bnow\b", r"\btrapped\b", r"\bbeing abused\b", r"\bin danger\b"],
    "harm_others": [r"\bkill someone\b", r"\bi will hurt\b"]
}

def get_helpline(country_code: Optional[str]=None) -> Dict[str,str]:
    if not country_code: return HELPLINES["DEFAULT"]
    cc = country_code.upper()
    return HELPLINES.get(cc, HELPLINES["DEFAULT"])

def detect_crisis(text: str) -> Tuple[bool,str,str]:
    if not text: return False, "low", "empty"
    t = text.lower()
    for p in CRISIS_KEYWORDS["suicide"] + CRISIS_KEYWORDS["self_harm"] + CRISIS_KEYWORDS["danger"]:
        if re.search(p, t, flags=re.IGNORECASE):
            return True, "high", f"matched:{p}"
    for p in CRISIS_KEYWORDS["harm_others"]:
        if re.search(p, t, flags=re.IGNORECASE):
            return True, "high", f"matched:{p}"
    return False, "low", "none"

def _format_helpline(h):
    lines = []
    if h.get("country"): lines.append(f"Country: {h['country']}")
    if h.get("emergency"): lines.append(f"Emergency: {h['emergency']}")
    if h.get("suicide_crisis"): lines.append(f"Suicide / Crisis: {h['suicide_crisis']}")
    for o in h.get("other", []): lines.append(f"- {o}")
    return "\n".join(lines)

def check_and_handle(text: str, user_meta: Dict[str,Any]=None, country_code: Optional[str]=None) -> Dict[str,Any]:
    is_crisis, level, reason = detect_crisis(text)
    helpline = get_helpline(country_code)
    if is_crisis:
        base = {
            "high": "If you are in immediate danger or thinking of harming yourself or others, please call emergency services right away.",
            "medium": "I am sorry you're in distress. Consider contacting a crisis line or a professional.",
            "low": "I hear you — if this gets worse, please seek immediate help."
        }.get(level, "")
        resp = f"{base}\n\nImmediate helpline info:\n{_format_helpline(helpline)}\n\nIf you'd like, I can find more local resources."
        # log
        rec = {"timestamp": datetime.utcnow().isoformat()+"Z", "text": text, "level": level, "reason": reason, "country_code": country_code}
        p = LOG_DIR / "flagged_messages.jsonl"
        with open(p, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"is_crisis": True, "level": level, "response": resp, "helpline": helpline, "reason": reason}
    return {"is_crisis": False, "level": level, "response": "", "helpline": helpline, "reason": reason}
