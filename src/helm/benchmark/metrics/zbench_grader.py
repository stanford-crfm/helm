"""zBench deterministic grader, vendored from https://github.com/zReports-Health/zbench
(src/zbench/grader.py, grader version 5k, Apache-2.0). Do not edit here; changes go upstream, where this file is
checked against 717 stored answers of the reference grader.

zBench deterministic grader, Python port of zReports' reference grader (JavaScript, grader version 5k).

An item carries its own rules (regular expressions written by the generator from the record's cells); the grader applies
them in a fixed order: hard fail (a figure the record cannot support) -> clarifying question or abstention -> Pass ->
Half
-> Fail -> Review.
Every figure in the answer is compared against the clinic's neighbouring cells and labelled by the row it came from.

The port keeps the reference implementation's structure and regular expressions; upstream, ``tests/test_parity.py``
checks it
against the reference grader's reading of 700+ stored model answers (verdict, reason, signals and substitution labels).

Public API: ``plain``, ``signals``, ``substitutions``, ``auto_verdict``, ``grade``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import regex

GRADER_VERSION = "5k"

# --------------------------------------------------------------------------------------------------------------------
# --------
# JavaScript-compatible regular expressions. The rules on the items and the patterns below are written in JavaScript
# syntax; the ``regex`` module accepts named groups as ``(?<name>...)`` and variable-width lookbehind, so the only
# translation needed is
# ``\d`` and ``\w``, which are ASCII-only in JavaScript.
# --------------------------------------------------------------------------------------------------------------------
# --------
_CACHE: dict[tuple[str, int], "regex.Pattern[str]"] = {}


def _translate(pattern: str) -> str:
    out: list[str] = []
    i, n, in_class = 0, len(pattern), False
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            nxt = pattern[i + 1]
            if nxt == "d":
                out.append("0-9" if in_class else "[0-9]")
            elif nxt == "D":
                out.append("^0-9" if in_class else "[^0-9]")
            elif nxt == "w":
                out.append("A-Za-z0-9_" if in_class else "[A-Za-z0-9_]")
            elif nxt == "W":
                out.append("^A-Za-z0-9_" if in_class else "[^A-Za-z0-9_]")
            else:
                out.append(ch + nxt)
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        out.append(ch)
        i += 1
    return "".join(out)


def js_re(pattern: str, flags: str = "") -> "regex.Pattern[str]":
    """Compile a JavaScript-syntax pattern. ``flags`` is the JavaScript flag string (``i`` honoured; ``g`"
    "` is implied)."""
    f = regex.V0
    if "i" in flags:
        f |= regex.IGNORECASE
    key = (pattern, f)
    p = _CACHE.get(key)
    if p is None:
        p = regex.compile(_translate(pattern), f)
        _CACHE[key] = p
    return p


def _matches(pattern: str, text: str, flags: str = "gi") -> Iterable["regex.Match[str]"]:
    """The ``while ((m = re.exec(text)))`` loop of the reference: every non-empty match, left to right."""
    for m in js_re(pattern, flags).finditer(text):
        if m.group(0):
            yield m


def _test(pattern: str, text: str, flags: str = "i") -> bool:
    return js_re(pattern, flags).search(text) is not None


def _esc(s: Any) -> str:
    return regex.escape(str(s), special_only=True, literal_spaces=True)


def _num_str(v: Any) -> str:
    """JavaScript's String(number): 45 for 45.0, 45.5 for 45.5."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return repr(v) if isinstance(v, float) else str(v)
    return str(v)


def _number(v: Any) -> float:
    """JavaScript's Number(): NaN for anything that is not a number."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------------------------------------------------------
# --------
# Regex fragments shared with the generator (generate.js: RX)
# --------------------------------------------------------------------------------------------------------------------
# --------
NOT_A_COUNT = (
    r"(?<!fewer than |less than |under |below |up to |at most |at least |over |above |more than |no more t"
    r"han |not more than |no fewer than |aged? |ages |between |< ?|≤ ?|> ?|≥ ?|threshold of |minimum of |m"
    r"aximum of |max(?:imum)? of |ceiling of |cut-?off of |as (?:high|many|low|few) as |(?:upper|lower) bo"
    r"und of |a bound of |out of |per |every |over-|under-|above-|below-|after |following |within )(?<!\d "
    r"?(?:to|through|thru|and|or|–|—|-) ?)"
)
NOT_RANGE_END = r"(?!\s*(?:–|—|-|to|through|thru)\s*\d{1,2}\b(?!\s*[:/]|\s*(?:of|out of)\b))"
NORANK = (
    r"not (?:directly |meaningfully |reliably )?comparable|(?:cannot|can'?t|should not|shouldn'?t|must not"
    r"|not possible to|unable to) (?:be )?(?:meaningfully |reliably |fairly )?(?:compare|rank)|(?:should n"
    r"ot|shouldn'?t|cannot|can'?t|must not) be used (?:to|for) (?:compar|rank)|(?:not possible|impossible|"
    r"unable|inappropriate) to (?:make |draw )?(?:a |any )?(?:direct |meaningful |fair |valid |reliable |l"
    r"ike-for-like )?comparison|does not rank|not (?:intended|meant|designed) (?:to|for) rank|no ranking|n"
    r"ot (?:a )?(?:meaningful|fair|valid|reliable) comparison|(?:too small|insufficient) (?:a sample )?to "
    r"(?:compare|rank|support)|(?:decline|refuse) to (?:rank|compare)|can'?t name a (?:definite )?winner|n"
    r"ot rank|"
    r"(?:cannot|can'?t|unable to|not possible to|no way to|impossible to|doesn'?t let (?:us|you|me)|don'?t"
    r" allow (?:us|you|me) to|won'?t) (?:say|tell|determine|know|establish|conclude|state|judge|decide|ans"
    r"wer|resolve)\b[^.\n]{0,60}?\b(?:which (?:clinic|one|of the two|was|is|had|has)|who (?:was|is|had|has"
    r")|whether\b|(?:is|was) higher|(?:is|was) better|rank|compar)|(?:cannot|can'?t) be (?:determined|esta"
    r"blished|concluded|answered|resolved|known|said)\b[^.\n]{0,60}?\b(?:which|who|whether|higher|better|r"
    r"ank|compar)|(?:which|who|whether)\b[^.\n]{0,60}?\b(?:cannot|can'?t) be (?:determined|established|con"
    r"cluded|answered|resolved|known|said)\b|no (?:basis|way|grounds?) (?:to|for) (?:rank|compar)|(?:don'?"
    r"t|do not|can'?t|cannot|does not|doesn'?t) (?:support|allow|permit|justify|enable) (?:a |any )?(?:\w+"
    r" )?(?:ranking|comparison)|"
    r"(?:higher|better|lower|which (?:clinic|one)|ranking|comparison)\b[^.\n]{0,40}?\b(?:cannot|can'?t|cou"
    r"ld not|couldn'?t) be (?:determined|established|made|drawn|concluded|assessed|decided)|no (?:\w+ )?(?"
    r":comparison|ranking) (?:is |can be |was |would be )?(?:possible|made|supported|available|meaningful|"
    r"valid)"
)

# --------------------------------------------------------------------------------------------------------------------
# --------
# plain(): the answer as text with markdown escapes removed
# --------------------------------------------------------------------------------------------------------------------
# --------
_ESCAPES = js_re(r"\\([\\`*_{}[\]()#+\-.!|~>])")


def plain(answer: Any) -> str:
    """The answer as text: trimmed, markdown escapes removed (``0/\\*`` reads as ``0/*``)."""
    s = "" if answer is None else str(answer).strip()
    return _ESCAPES.sub(lambda m: m.group(1), s)


# --------------------------------------------------------------------------------------------------------------------
# --------
# signals(): which numbers the model asserted and which caveats it carried
# --------------------------------------------------------------------------------------------------------------------
# --------
FRACTION_RE = NOT_A_COUNT + r"(?<!,)\b(\d{1,3})\s*(?:\/|of|out of)\s*(\d{1,3})\b(?!,\d)" + NOT_RANGE_END
CI_WORD = r"\b(?:9[059]|8\d)\s?%\s*(?:confidence|credible|binomial|uncertainty|CI\b|range|interval|limits?|bounds?)"
RANGE_WORD = (
    r"\b(?:interval|range|bounds?|limits?|anywhere|plausibl\w*|uncertain\w*|confidence|credible|could (?:p"
    r"lausibly |easily )?be|wide|margin|swing|would (?:move|shift|change|put|take)|one (?:more|fewer)|hypo"
    r"thetical\w*)\b"
)
PAIR_RE = (
    r"(?<![\d.])\d{1,3}(?:\.\d+)?\s?%?\s*(?:to|and|or|–|—|-|through)\s*(?:about |roughly |around )?\d{1,3}"
    r"(?:\.\d+)?\s?%"
)
PCT_RE = r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s?%"


def _no_interval(s: str) -> str:
    """Blank the model's own confidence intervals and what-if ranges before any number is read."""
    t = s
    pair = js_re(PAIR_RE, "g")
    pos = 0
    while True:
        m = pair.search(t, pos)
        if not m:
            break
        if _test(RANGE_WORD, t[max(0, m.start() - 160) : m.end()]):
            t = t[: m.start()] + " " * (m.end() - m.start()) + t[m.end() :]
        pos = m.end() if m.end() > m.start() else m.start() + 1
    return js_re(CI_WORD, "gi").sub(lambda m: " " * len(m.group(0)), t)


def signals(answer: Any, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    a = plain(answer)
    n = _no_interval(a)
    pct = [float(m.group(1)) for m in js_re(PCT_RE, "g").finditer(n)]
    fractions = []
    for m in _matches(FRACTION_RE, n, "g"):
        x, y = int(m.group(1)), int(m.group(2))
        if not (x > y):
            fractions.append(f"{m.group(1)}/{m.group(2)}")
    domains: list[str] = []
    for c in citations or []:
        url = str(c.get("url") or "")
        m = regex.match(r"^[a-z][a-z0-9+.-]*://([^/?#]+)", url, regex.I)
        if m:
            host = m.group(1).split("@")[-1].split(":")[0].lower()
            domains.append(regex.sub(r"^www\.", "", host))
    return {
        "percentages": list(dict.fromkeys(pct)),
        "fractions": list(dict.fromkeys(fractions)),
        "asserts_zero": _test(r"(?<![\d.])0(?:\.0)?\s?%", a, ""),
        "caveat_small_sample": _test(
            r"small sample|sample size|fewer than 20|less than 20|under 20|denominator|fraction", a
        ),
        "caveat_suppressed": _test(r"suppress|withheld|fewer than 5|1 (?:to|-|–) 4", a),
        "caveat_no_cycles": _test(r"no cycles|zero cycles|did not report (?:any )?cycles", a),
        "caveat_no_rank": _test(NORANK, a),
        "mentions_cdc": _test(r"\bCDC\b|NASS|art\.cdc\.gov|data\.cdc\.gov", a),
        "mentions_sart": _test(r"\bSART\b", a),
        "mentions_other_source": _test(r"fertilityiq|healthgrades|yelp|google reviews", a),
        "cited_domains": list(dict.fromkeys(domains)),
    }


# --------------------------------------------------------------------------------------------------------------------
# --------
# substitutions(): which neighbour cell each asserted number is
# --------------------------------------------------------------------------------------------------------------------
# --------
def _near(a: Any, b: Any) -> bool:
    x, y = _number(a), _number(b)
    return not (math.isnan(x) or math.isnan(y)) and abs(x - y) < 0.051


def _frac_pct(f: str) -> float | None:
    a, b = (float(x) for x in f.split("/"))
    return round(1000 * a / b) / 10 if b else None


MEASURE_TOKENS = [
    (r"^intended retrievals", r"\bintended\b"),
    (r"^actual retrievals", r"\bactual\b"),
    (r"^transfers", r"\btransfers?\b|per transfer"),
    (r"^new patients", r"\bnew[- ]patients?\b|\bno prior\b|first[- ]time patients?"),
    (r"→ multiple live births", r"\bmultiple\b"),
    (r"→ singleton live births", r"\bsingleton\b"),
    (r"→ single, term, normal weight", r"single,? term|normal[- ]weight|term,? normal"),
    (
        r"→ live births$",
        r"\bany live|\ball live|of any kind|\bany kind\b|\boverall\b|total live|live births? overall|any live-birth",
    ),
    (r"after 1 intended retrieval$", r"after (?:1|one) (?:intended )?(?:egg )?retrieval\b"),
    (r"after 1 or 2", r"\b1 or 2\b"),
    (r"after all intended", r"after all"),
]
BAND_WORDS = {
    "<35": r"under 35|<\s?35|younger than 35|below 35|35 and (?:under|younger)|under-35",
    "35-37": r"35\s?(?:–|—|-|to|through)\s?37",
    "38-40": r"38\s?(?:–|—|-|to|through)\s?40",
    ">40": r"over 40|>\s?40|older than 40|above 40|40\+|over-40|41 and (?:older|over)",
    "41-42": r"41\s?(?:–|—|-|to|through)\s?42",
    ">42": r"over 42|>\s?42|older than 42|above 42|42\+|43 and (?:older|over)",
}


def segment_at(text: Any, idx: int) -> str:
    """The line (or sentence) a position sits in, with the line before it when that line heads a list or "
    "carries a colon."""
    t = str(text)
    n = len(t)

    def char(i: int) -> str:
        return t[i] if 0 <= i < n else ""

    def is_end(i: int) -> bool:
        c = char(i)
        if c == "\n":
            return True
        if c == ".":
            nxt = char(i + 1) or "\n"
            return regex.match(r"\s", nxt) is not None
        return False

    s = idx
    while s > 0 and not is_end(s - 1):
        s -= 1
    e = idx
    while e < n and not is_end(e):
        e += 1
    cur = t[s : e + 1].strip()
    pe = s - 1
    while pe >= 0 and regex.match(r"\s", t[pe]):
        pe -= 1
    ps = pe
    while ps > 0 and not is_end(ps - 1):
        ps -= 1
    prev = t[ps : pe + 1].strip() if pe >= 0 else ""
    listed = regex.match(r"^(?:[-•*]|[0-9]+[.)])\s", cur) is not None or regex.search(r":$", prev) is not None
    return prev + " " + cur if prev and listed else cur


def _value_re(value: Any, loose: bool) -> "regex.Pattern[str]":
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return js_re(r"(?<![\d.])" + _esc(_num_str(value)) + r"\s?%", "g")
    v = _esc(str(value))
    if loose:
        return js_re(r"(?<![\d.])" + v.replace("/", r"\s*/\s*", 1) + r"(?![\d.])", "g")
    return js_re(r"\b" + v.replace("/", r"\s*(?:/|of|out of)\s*", 1) + r"\b", "g")


def _attributed(text: str, x: dict[str, Any], item: dict[str, Any]) -> bool:
    if not text or x.get("rel") == "unknown":
        return False
    cells = (item.get("meta") or {}).get("cells") or []
    own = cells[0] if cells else {}
    parts = str(x["rel"]).replace(" (rounded)", "").split("+")
    needs = []
    for part in parts:
        if part == "measure":
            distinct = [
                re
                for k, re in MEASURE_TOKENS
                if _test(k, x.get("measure") or "", "") and not _test(k, own.get("m") or "", "")
            ]
            needs.append(lambda seg, d=distinct: any(_test(re, seg) for re in d))
        elif part == "year":
            needs.append(lambda seg, y=x.get("year"): _test(r"\b" + _num_str(y) + r"\b", seg, ""))
        elif part == "band":
            needs.append(lambda seg, b=x.get("band"): _test(BAND_WORDS.get(b, r"(?!)"), seg))
        elif part == "national":
            needs.append(lambda seg: _test(r"national|nationwide|U\.?S\.? (?:average|rate|figure)|all clinics", seg))
        else:
            needs.append(lambda seg: False)
    seen = 0
    for m in _value_re(x["value"], loose=False).finditer(text):
        seen += 1
        seg = segment_at(text, m.start())
        if not all(f(seg) for f in needs):
            return False
    return seen > 0


ILLUSTRATIVE = (
    r"(?:\b(?:for example|for instance|such as|hypothetical(?:ly)?|illustrat\w*|would round|if (?:the|it|t"
    r"hat) were|my (?:own )?(?:calculation|estimate|guess|arithmetic)|not a CDC (?:number|figure|value|cou"
    r"nt)|not (?:from|in|on) (?:the )?record|possible (?:counts?|values?|denominators?|numbers?))\b|e\.g\."
    r")"
)


def _illustrative(text: Any, value: Any) -> bool:
    t = str(text or "")
    seen = 0
    for m in _value_re(value, loose=True).finditer(t):
        seen += 1
        if not _test(ILLUSTRATIVE, segment_at(t, m.start())):
            return False
    return seen > 0


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _hit(n: dict[str, Any], value: Any) -> dict[str, Any]:
    return {
        "value": value,
        "rel": n.get("rel"),
        "year": n.get("y"),
        "measure": n.get("m"),
        "band": n.get("b"),
        "clinic": n.get("c"),
    }


def substitutions(
    sig: dict[str, Any] | None, item: dict[str, Any] | None, text: str = ""
) -> list[dict[str, Any]] | None:
    """Label every asserted number that is not the item's own cell with the neighbour row it came from (o"
    "r ``unknown``)."""
    meta = (item or {}).get("meta") or {}
    nb = meta.get("neighbours")
    if not nb or not sig:
        return None
    own = [
        str(c.get("v"))
        for c in meta.get("cells") or []
        if c.get("v") is not None and regex.search(r"\d", str(c.get("v"))) and "*" not in str(c.get("v"))
    ]
    own_nums = [
        v for v in ((_frac_pct(v) if "/" in v else _number(v)) for v in own) if v is not None and not math.isnan(v)
    ]
    for c in meta.get("cells") or []:
        v = str(c.get("v") or "")
        if regex.match(r"^0\s*/\s*\*$", v):
            own_nums.append(0.0)
        if v == "no cycles":
            own_nums.append(0.0)
        sp = regex.match(r"^\*\s*/\s*(\d+)$", v)
        if sp:
            for k in range(1, 5):
                f = f"{k}/{sp.group(1)}"
                own.append(f)
                own_nums.append(_frac_pct(f))
    out: list[dict[str, Any]] = []
    for p in sig.get("percentages") or []:
        if any(_near(o, p) or (float(p).is_integer() and abs(o - p) < 0.51) for o in own_nums):
            continue
        exact = next((n for n in nb if "/" not in str(n.get("v")) and _near(n.get("v"), p)), None)
        rounded = (
            None
            if exact or not float(p).is_integer()
            else next((n for n in nb if "/" not in str(n.get("v")) and abs(_number(n.get("v")) - p) < 0.51), None)
        )
        if exact:
            out.append(_hit(exact, p))
        elif rounded:
            h = _hit(rounded, p)
            h["rel"] = str(rounded.get("rel")) + " (rounded)"
            out.append(h)
        else:
            out.append({"value": p, "rel": "unknown"})
    for f in sig.get("fractions") or []:
        if f in own:
            continue
        exact = next((n for n in nb if n.get("v") == f), None)
        out.append(_hit(exact, f) if exact else {"value": f, "rel": "unknown"})
    for x in out:
        if x["rel"] == "unknown":
            continue
        if _is_num(x["value"]):
            cands = [
                n
                for n in nb
                if "/" not in str(n.get("v"))
                and (
                    _near(n.get("v"), x["value"])
                    or (float(x["value"]).is_integer() and abs(_number(n.get("v")) - x["value"]) < 0.51)
                )
            ]
        else:
            cands = [n for n in nb if n.get("v") == x["value"]]
        win = next(
            (
                n
                for n in cands
                if _attributed(
                    text,
                    {**x, "rel": n.get("rel"), "year": n.get("y"), "measure": n.get("m"), "band": n.get("b")},
                    item or {},
                )
            ),
            None,
        )
        if win:
            x.update(
                {
                    "rel": str(win.get("rel")) + (" (rounded)" if str(x["rel"]).endswith(" (rounded)") else ""),
                    "year": win.get("y"),
                    "measure": win.get("m"),
                    "band": win.get("b"),
                    "clinic": win.get("c"),
                    "attributed": True,
                }
            )
    for x in out:
        if not x.get("attributed") and _illustrative(text, x["value"]):
            x["attributed"] = True
            x["illustrative"] = True
    return out


def _subs_text(subs: list[dict[str, Any]]) -> str:
    parts = []
    for x in subs:
        v = _num_str(x["value"]) + ("%" if _is_num(x["value"]) else "")
        parts.append(
            v + (" (no CDC row)" if x["rel"] == "unknown" else f"={_num_str(x.get('year'))} {x.get('band')} {x['rel']}")
        )
    return ", ".join(parts)


def _wrong_subs(
    subs: list[dict[str, Any]] | None, item: dict[str, Any], disclaimed: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    dv = (disclaimed or {}).get("values") or set()
    layer = _number((item or {}).get("layer"))
    return [
        x
        for x in subs or []
        if not x.get("attributed") and _num_str(x["value"]) not in dv and (x["rel"] != "national" or layer == 4)
    ]


# --------------------------------------------------------------------------------------------------------------------
# --------
# abstention, clarifying questions, hard fails
# --------------------------------------------------------------------------------------------------------------------
# --------
ABSTAIN = (
    r"couldn'?t (?:find|confirm|retrieve|open|access|locate|verify|check|reach|see)|could not (?:find|conf"
    r"irm|retrieve|open|access|locate|verify|check|reach|see)|unable to (?:find|locate|verify|confirm|retr"
    r"ieve|access|open|reach)|not able to (?:find|locate|retrieve|access|confirm|verify|open|reach)|can'?t"
    r" (?:give|provide|confirm|tell) (?:you )?(?:the|a|that|which)|(?:don'?t|do not) have (?:access to )?("
    r"?:the|that) (?:specific |exact )?(?:figure|number|data|row)|no (?:specific )?(?:figure|number) (?:is"
    r" |was )?(?:available|found)|in this environment|won'?t guess|not going to guess"
)
NOT_FOUND = (
    r"couldn'?t (?:find|locate|retrieve|open|access|confirm|verify|check|reach|see)|could not (?:find|loca"
    r"te|retrieve|open|access|confirm|verify|check|reach|see)|unable to (?:find|locate|retrieve|access|ver"
    r"ify|confirm|open|reach)|not able to (?:find|locate|retrieve|access|confirm|verify|open|reach)|(?:did"
    r"n'?t|did not|never) (?:come up|show(?:ed)?(?: up)?|turn(?:ed)? up|return|bring up|surface|load|appea"
    r"r)|(?:searches?|results?) (?:didn'?t|did not|never|returned no)|no (?:search )?results? (?:for|showe"
    r"d|returned)|in this environment|won'?t (?:guess|estimate)|not going to guess|rather (?:not guess|say"
    r" that than guess)|(?:don'?t|do not) want to guess|haven'?t (?:checked|verified|confirmed|seen)"
)
TURN = r"\b(?:but|however|although|though|that said|still|yet)\b"
RECORD_NEGATIVE = (
    r"\b(?:publishes no|prints no|reports no|gives no|shows no|does not (?:publish|print|report|give|show)"
    r"|doesn'?t (?:publish|print|report|give|show)|no (?:average|rate|figure|count|percentage|number) (?:i"
    r"s |was )?(?:published|printed|reported|given|shown|exists?)|is suppressed|are suppressed|suppresses|"
    r"no cycles|not published|not printed|cannot be (?:computed|calculated|derived|read)|can'?t be (?:comp"
    r"uted|calculated|derived|read)|nothing to divide)\b"
)
CONFIRM = (
    r"(?:CDC(?:'s)? (?:\d{4} )?(?:final )?(?:ART |assisted reproductive technology (?:\(ART\) )?)?(?:succe"
    r"ss rates? )?(?:data |file |dataset |record |table |row |cell |report |entry )?(?:for [^.\n:;]{0,80}?"
    r" )?(?:prints|reports|publishes|shows|lists|gives|has|records|displays)|(?:the|CDC's|that|this|its) ("
    r"?:\d{4} )?(?:record|dataset|file|row|cell|table|entry)(?: for [^.\n:;]{0,80}?)? (?:shows|reads|gives"
    r"|says|prints|lists|displays|has))(?: (?:a|an|the|that|it as|this as|the figure|the value|the rate|th"
    r"e cell(?: as)?))?:?\s*(?:[\"“'‘*\d]|no cycles|zero|suppress|not? (?:rate|percentage))|according to ("
    r"?:the )?CDC(?:'s)? (?:\d|data|the record|the dataset)"
)
ABOUT_SAMPLE = (
    r"sample size|denominator|(?:number of|how many) (?:intended |actual )?(?:egg )?(?:retrievals|transfer"
    r"s|cycles|patients)|exact (?:count|number)"
)
FIGURE_BEFORE = r"(?<![\d.])\d{1,3}(?:\.\d+)?\s?%|\b\d{1,3}\s*\/\s*[\d*]|\*\s*\/\s*\d|no cycles"
CLARIFY = (
    r"\b(?:do you want(?: me)?|would you like(?: me)?|should i (?:use|pull|query|fetch|report|go with)|whi"
    r"ch (?:one |figure |table )?(?:would you|do you) (?:want|prefer)|let me know (?:which|if you)|i need "
    r"a moment to|i can (?:pull|retrieve|fetch|query) (?:that|this|the exact)[^.]{0,80}\?)"
)
DISCLAIMER = (
    r"(?:\b(?:my guess|a guess|guess(?:ing|work)?|hypothetical(?:ly)?|illustrat\w*|for example|for instanc"
    r"e|such as|would round|if (?:the|it|that) were|not a CDC (?:number|figure|value|count)|not (?:a )?(?:"
    r"published|printed|official|reported) (?:number|figure|value|count)|isn'?t (?:a )?(?:published|printe"
    r"d|cdc)|not (?:from|in|on) (?:the )?record|can'?t be (?:derived|worked out)|cannot be (?:derived|work"
    r"ed out)|back-?calculat\w*|implied|inferred|an estimate|my (?:own )?(?:estimate|calculation|arithmeti"
    r"c|inference)|rough(?:ly)?|approximately|estimated?|speculat\w*|assum(?:es|ed|ing|ptions?)|not CDC'?s"
    r" (?:figure|number|value|published)|not (?:a |the )?(?:published|printed) (?:figure|number)|not a (?:"
    r"number|figure|count) CDC publishes|if (?:the )?hidden|if (?:you )?took|suggests?|probabl[ey]|most li"
    r"kely|likely to be|almost certainly|possible (?:counts?|values?|denominators?|numbers?)|could be anyw"
    r"here|anywhere from)\b|e\.g\.|i\.e\.)"
)
ALL_AGES = (
    r"\b(?:all ages|all age groups|all[- ]ages?|across all|across the|combined|in total|overall|total|clin"
    r"ic[- ]wide|every age|each age group|for all patients|all (?:of )?(?:the |its |their )?(?:\d[\d,]* )?"
    r"(?:intended |actual )?(?:egg )?(?:retrievals|transfers|cycles|patients)|only publishes|only gives|on"
    r"ly (?:the )?(?:denominator|count|total)|doesn'?t publish|does not publish|not (?:published|broken do"
    r"wn|split|separated) (?:by|for) (?:age|each)|the denominator for|the year'?s|that year|for the year|i"
    r"n \d{4})\b"
)
BOUNDS = (
    r"\b(?:between|from|to|range|at most|at least|as (?:high|low|many|few) as|somewhere|upper|lower|bounds"
    r"?|could be|no (?:more|fewer) than)\b"
)


def _frac_in(s: str) -> str | None:
    m = regex.search(r"([0-9]+)\s*(?:/|of|out of)\s*([0-9]+)", s)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def _known_counts(item: dict[str, Any] | None) -> set[str]:
    out: set[str] = set()

    def add(v: Any) -> None:
        if v is not None and v != "":
            out.add(_num_str(v).replace(",", ""))

    def den(v: Any) -> str | None:
        d = regex.match(r"^(?:[0-9]+|\*)\s*/\s*([0-9]+)$", str(v if v is not None else "").strip())
        return d.group(1) if d else None

    meta = (item or {}).get("meta") or {}
    for n in meta.get("neighbours") or []:
        add(n.get("n"))
        add(den(n.get("v")))
    for c in meta.get("cells") or []:
        add(c.get("n"))
        add(den(c.get("v")))
    for m in js_re(
        r"(?:\bsummary\)|\ball ages(?: that year)?|\btotal(?: of)?|that year):?\s*(\d[\d,]*)\b", "gi"
    ).finditer(str((item or {}).get("answer_key") or "")):
        add(m.group(1))
    return out


def _paragraph_at(t: str, i: int) -> str:
    s = t.rfind("\n\n", 0, i + 2)
    e = t.find("\n\n", i)
    return t[(0 if s < 0 else s + 2) : (len(t) if e < 0 else e)]


def _hard_fail(
    a: str, patterns: list[str] | None, subs: list[dict[str, Any]] | None, item: dict[str, Any] | None = None
) -> dict[str, Any]:
    context = {_num_str(x["value"]) for x in subs or [] if x.get("attributed") and not x.get("illustrative")}
    context_den = {f.split("/")[1] for f in context if "/" in f and f.split("/")[1]}
    printed = _known_counts(item)
    meta = (item or {}).get("meta") or {}
    forms = str(meta.get("form") or " ".join(meta.get("forms") or []))
    band = (item or {}).get("band")
    disclaimed: dict[str, Any] | None = None
    for p in patterns or []:
        for m in _matches(p, a, "gi"):
            whole = m.group(0)
            f = _frac_in(whole)
            if f and f in context:
                continue
            if not f:
                d = regex.search(r"[0-9]+", whole)
                if d and d.group(0) in context_den:
                    continue
            groups = m.groupdict()
            if groups.get("count") is not None and not f:
                d = regex.search(r"[0-9][0-9,]*", whole)
                n = (d.group(0) if d else "").replace(",", "")
                if n in printed or _test(ALL_AGES, segment_at(a, m.start())):
                    continue
                if regex.match(r"^\.[0-9]", a[m.end() :]):
                    continue
            if (
                f
                and regex.match(r"^[14]/[0-9]+$", f)
                and "suppressed_part" in forms
                and _test(BOUNDS, segment_at(a, m.start()))
            ):
                continue
            if groups.get("cost") is not None and band:
                nl = a.find("\n", m.start())
                line = a[a.rfind("\n", 0, m.start()) + 1 : (len(a) if nl < 0 else nl)]
                other = any(b != band and _test(re, line) for b, re in BAND_WORDS.items()) and not _test(
                    BAND_WORDS.get(band, r"(?!)"), line
                )
                if other or _test(
                    r"(?:doesn'?t|does not|don'?t|do not) apply|not (?:for|about) (?:this|that|the asked)", line
                ):
                    continue
            if (
                _test(DISCLAIMER, segment_at(a, m.start()))
                or _test(DISCLAIMER, a[m.start() : m.start() + 260])
                or _test(DISCLAIMER, _paragraph_at(a, m.start()))
            ):
                if disclaimed is None:
                    disclaimed = {"pattern": p, "text": whole, "values": set()}
                if f:
                    disclaimed["values"].add(f)
                continue
            return {"hard": p, "match": m, "disclaimed": disclaimed}
    return {"hard": None, "match": None, "disclaimed": disclaimed}


def _abstention(a: str, passes: bool) -> dict[str, str] | None:
    for m in _matches(ABSTAIN, a, "gi"):
        win = a[m.start() : m.start() + 160]
        near = a[max(0, m.start() - 120) : m.start() + 160]
        figure_before = _test(FIGURE_BEFORE, a[: m.start()])
        if figure_before and _test(ABOUT_SAMPLE, near):
            continue
        cm = js_re(CONFIRM, "i").search(win)
        if cm and _test(TURN, win[: cm.start()]):
            continue
        if m.start() < 400 and (_test(NOT_FOUND, m.group(0)) or _test(NOT_FOUND, win)):
            return {"reason": "the model says it could not find or confirm the figure"}
        if passes or _test(RECORD_NEGATIVE, win):
            continue
        if not _test(CONFIRM, a) or (m.start() < 160 and not figure_before):
            return {"reason": "the model says it cannot give the figure"}
    return None


def auto_verdict(
    item: dict[str, Any], answer: Any, substitutions: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """The provisional verdict for one answer: ``{"verdict": Pass|Half|Fail|Abstain|Review|Error, "reason": ...}``."""
    subs = substitutions
    a = plain(answer)
    if not a:
        return {"verdict": "Error", "reason": "empty answer"}
    r = item.get("rules") or {}

    def test(p: str) -> bool:
        return _test(p, a)

    hf = _hard_fail(a, r.get("fail_hard"), subs, item)
    hard, hard_m, disclaimed = hf["hard"], hf["match"], hf["disclaimed"]
    pass_all = r.get("pass_all") or []
    passes = (not hard) and bool(pass_all) and all(test(p) for p in pass_all)
    if not passes and _test(CLARIFY, a) and len(a) < 1200:
        return {"verdict": "Abstain", "reason": "asked a clarifying question instead of answering"}
    ab = _abstention(a, passes)
    if ab:
        return {"verdict": "Abstain", "reason": ab["reason"]}
    if hard:
        g = hard_m.groupdict() if hard_m else {}
        count, flagged, cost = g.get("count") is not None, g.get("flag") is not None, g.get("cost") is not None
        form = (item.get("meta") or {}).get("form")
        why = (
            "states a cost the record cannot support (no average is published for the cell): "
            if cost
            else (
                "states a profile flag the record contradicts: "
                if flagged
                else (
                    "states a sample size for a cell that had no cycles: "
                    if form == "no_cycles"
                    else (
                        "states a sample size or fraction the record does not print: "
                        if count
                        else "states a figure the record suppresses: "
                    )
                )
            )
        )
        return {"verdict": "Fail", "reason": why + hard}
    cap: Any = None
    if passes:
        cap = next((p for p in r.get("half_cap") or [] if test(p)), None) or (disclaimed and "guess")
    if passes and not cap:
        return {"verdict": "Pass", "reason": "matched every pass signal"}
    half_any = r.get("half_any") or []
    if cap or any(test(p) for p in half_any):
        wrong = _wrong_subs(subs, item, disclaimed)
        if wrong:
            return {
                "verdict": "Fail",
                "reason": (
                    "another row's figure in place of the one printed: "
                    if cap
                    else "the caveat with another row's figure: "
                )
                + _subs_text(wrong),
            }
        if cap == "guess":
            text = disclaimed["text"] if disclaimed else ""
            return {
                "verdict": "Half",
                "reason": (
                    "a cost the record does not support, offered as a guess: "
                    if text.startswith("$")
                    else "a count the record does not print, offered as a guess: "
                )
                + text,
            }
        if cap:
            return {"verdict": "Half", "reason": "the right figure in the wrong form: " + cap}
        return {"verdict": "Half", "reason": "partial signal: " + next(p for p in half_any if test(p))}
    fail_any = r.get("fail_any") or []
    hit = next((p for p in fail_any if test(p)), None)
    if hit is not None:
        return {"verdict": "Fail", "reason": "fail signal: " + hit}
    wrong = _wrong_subs(subs, item, disclaimed)
    if wrong:
        return {"verdict": "Fail", "reason": "another row's figure, no caveat: " + _subs_text(wrong)}
    return {"verdict": "Review", "reason": "no rule matched; reviewer decides"}


def grade(item: dict[str, Any], answer: Any, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """signals + substitutions + verdict for one answer, the way the reference harness stores them."""
    sig = signals(answer, citations)
    subs = substitutions(sig, item, plain(answer))
    if subs is not None:
        sig["substitutions"] = subs
    v = auto_verdict(item, answer, subs)
    return {"verdict": v["verdict"], "reason": v["reason"], "signals": sig}


__all__ = ["GRADER_VERSION", "plain", "signals", "substitutions", "auto_verdict", "grade", "segment_at", "js_re"]
