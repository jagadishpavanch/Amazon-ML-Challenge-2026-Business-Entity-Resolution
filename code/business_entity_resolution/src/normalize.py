"""Name / address normalisation and light parsing (country-agnostic).

Every record gets the same treatment regardless of its country label; the rules
cover US, Indian and French conventions at once.
"""
import os
import re
from multiprocessing import Pool

import jellyfish
import pandas as pd
from unidecode import unidecode

# ----------------------------------------------------------------------------- names
NAME_ABBR = {
    "corp": "corporation", "intl": "international", "mfg": "manufacturing", "co": "company",
    "bros": "brothers", "svc": "services", "svcs": "services", "mgmt": "management",
    "assoc": "associates", "dept": "department", "natl": "national", "ent": "enterprises",
    "pvt": "private", "ltd": "limited", "inc": "incorporated", "mfrs": "manufacturers",
    "engg": "engineering", "eng": "engineering", "cons": "consultants", "govt": "government",
    "hosp": "hospital", "univ": "university", "inst": "institute", "assn": "association",
    "ctr": "center", "centre": "center", "grp": "group", "hldgs": "holdings", "ind": "industries",
    "inds": "industries", "st": "saint", "ste": "sainte", "cie": "compagnie", "ste.": "societe",
    "soc": "societe", "sté": "societe", "tech": "technologies",
}
EXTRA_ABBR = {}  # filled by mine_abbreviations() on train, then persisted with the model

# multi-token legal forms are matched first (on the space-joined token string)
LEGAL_MULTI = [
    ("private limited", "PVT"), ("pvt limited", "PVT"), ("pvt ltd", "PVT"), ("p ltd", "PVT"),
    ("pte ltd", "PVT"), ("et compagnie", "ETCIE"), ("et cie", "ETCIE"), ("and company", "CO"),
    ("and fils", "ETFILS"), ("et fils", "ETFILS"), ("limited liability company", "LLC"),
    ("professional corporation", "PC"),
]
LEGAL_SINGLE = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "llc": "LLC",
    "ltd": "LTD", "limited": "LTD", "pvt": "PVT", "private": "PVT", "llp": "LLP", "plc": "PLC",
    "lp": "LP", "pllc": "PLLC", "pc": "PC", "pa": "PA", "co": "CO", "company": "CO",
    "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "sa": "SA", "eurl": "EURL", "snc": "SNC",
    "sci": "SCI", "scop": "SCOP", "ei": "EI", "eirl": "EI", "selarl": "SELARL", "selas": "SELAS",
    "gie": "GIE", "sca": "SCA", "scs": "SCS", "scm": "SCM", "scp": "SCP", "gmbh": "GMBH", "opc": "OPC",
}
NAME_STOP = {"the", "of", "and", "et", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an", "dr", "m/s", "ms"}
DBA_SPLIT = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|t\s*/\s*a|trading as|aka|a\.k\.a\.?)\b")
PAREN = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
WEB = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|co|us|io)\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
APOS = re.compile(r"['`’]")
ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|e|eme|ere)\b")


def _basic(s: str) -> str:
    s = unidecode(s).lower()
    s = s.replace("&", " and ").replace("+", " and ")
    s = APOS.sub("", s)
    return s


def _tokens(s: str):
    return NON_ALNUM.sub(" ", s).split()


def _legal_and_core(tokens):
    """Return (legal category, core tokens) for a token list."""
    joined = " " + " ".join(tokens) + " "
    legal = ""
    for pat, cat in LEGAL_MULTI:
        if f" {pat} " in joined:
            legal = legal or cat
            joined = joined.replace(f" {pat} ", " ")
    toks = joined.split()
    core = []
    for t in toks:
        if t in LEGAL_SINGLE:
            legal = legal or LEGAL_SINGLE[t]
            continue
        if t in NAME_STOP:
            continue
        core.append(EXTRA_ABBR.get(t, NAME_ABBR.get(t, t)))
    return legal, core


def _phon(tokens):
    out = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
        else:
            try:
                out.append(jellyfish.metaphone(t) or t)
            except Exception:
                out.append(t)
    return out


_REPEAT = re.compile(r"(.)\1+")
_VOWELS = re.compile(r"(?<=.)[aeiouyh]")


def skeleton(tok: str) -> str:
    """Consonant skeleton: collapse repeats, drop non-initial vowels/h ('praaivett' -> 'prvt')."""
    return _REPEAT.sub(r"\1", _VOWELS.sub("", _REPEAT.sub(r"\1", tok)))


def norm_name(raw: str) -> dict:
    b = _basic(raw)
    web = WEB.match(b.strip().lstrip("#@ "))
    variants = []
    m = DBA_SPLIT.split(b)
    if len(m) > 1:
        variants.extend(p for p in m if p.strip())
    paren = PAREN.findall(b)
    no_paren = PAREN.sub(" ", b)
    if paren and no_paren.strip():
        variants.append(no_paren)
    toks = _tokens(b)
    name_norm = " ".join(toks)
    legal, core = _legal_and_core(toks)
    if web:  # website-style name: keep the stem as the core
        core = [web.group(1).replace("-", "")]
    core_s = " ".join(core) if core else name_norm
    var_cores = []
    for v in variants:
        _, vc = _legal_and_core(_tokens(v))
        if vc and " ".join(vc) != core_s:
            var_cores.append(" ".join(vc))
    ctoks = core_s.split()
    return dict(
        name_norm=name_norm,
        name_core=core_s,
        name_compact=core_s.replace(" ", ""),
        name_sorted=" ".join(sorted(ctoks)),
        legal=legal,
        name_vars="|".join(var_cores),
        acronym="".join(t[0] for t in ctoks if t) if len(ctoks) > 1 else "",
        name_phon=" ".join(_phon(ctoks)),
        name_skel=" ".join(skeleton(t) for t in name_norm.split()),
        name_nums=" ".join(t.lstrip("0") or "0" for t in ctoks if t.isdigit()),
        is_web=int(bool(web) or b.strip().startswith("#")),
        nonlatin=int(any(ord(ch) > 0x24F for ch in raw)),
    )


# -------------------------------------------------------------------------- addresses
ADDR_ABBR = {
    "rd": "road", "st": "st", "str": "st", "street": "st", "saint": "st", "sainte": "ste", "suite": "ste",
    "r": "rue", "che": "chemin", "rn": "route", "qt": "quartier", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway", "ste": "ste",
    "fl": "floor", "flr": "floor", "apt": "apartment", "bldg": "building", "nr": "near",
    "opp": "opposite", "sec": "sector", "ph": "phase", "chem": "chemin", "pl": "place",
    "fg": "faubourg", "ct": "court", "cir": "circle", "pkwy": "parkway", "ter": "terrace",
    "sq": "square", "trl": "trail", "pt": "point", "mt": "mount", "ft": "flat", "hno": "house",
    "no": "number", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest", "crs": "cross", "ngr": "nagar",
    "clny": "colony", "mkt": "market", "stn": "station", "dist": "district", "tq": "taluk",
    "po": "post", "pkg": "parking", "imp": "impasse", "all": "allee", "rte": "route", "qu": "quai",
    "unit": "unit", "bldng": "building", "cplx": "complex", "comp": "complex", "hsg": "housing",
    "soc": "society", "ind": "industrial", "estt": "estate", "est": "estate", "bazar": "bazaar",
}
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3",
}
ADDR_DROP = {"null", "none", "na", "nan", "unit", "suite", "apartment", "number", "house", "box",
             "post", "office", "of", "the", "and", "de", "du", "des", "la", "le", "les", "d", "l"}
# Canonicalise state / region codes and names (static text normalisation, no lookup).
STATES = {
    # US
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    # India
    "ap": "andhra pradesh", "arp": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ct.": "chhattisgarh", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "keralam": "kerala", "mp": "madhya pradesh", "mh": "maharashtra",
    "mn.": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or.": "odisha",
    "orissa": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn.": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "uttaranchal": "uttarakhand", "wb": "west bengal", "dl": "delhi", "jk": "jammu and kashmir",
    "ch": "chandigarh", "py": "puducherry", "pondicherry": "puducherry", "ga.": "goa",
}
POST6 = re.compile(r"\b\d{6}\b")
POST5 = re.compile(r"\b\d{5}(?:-\d{4})?\b")
LANDMARK = re.compile(r"\b(?:near|nr|opposite|opp|behind|beside|next to|a cote de|pres de|en face de)\b\.?\s*([^,]*)")


NUM_SIGN = re.compile(r"\b[Nn]\s*[°º]\s*")


def norm_addr(raw: str) -> dict:
    b = unidecode(NUM_SIGN.sub("no ", raw).replace("°", " ").replace("º", " ")).lower()
    b = ORDINAL.sub(r"\1", b)
    landmark = " ".join(_tokens(" ".join(LANDMARK.findall(b))))
    b_core = LANDMARK.sub(" ", b)
    comps = [c.strip() for c in b_core.split(",") if c.strip()]
    pcs = POST6.findall(b) or POST5.findall(b)
    postcode = pcs[-1].split("-")[0] if pcs else ""
    toks = []
    comp_toks = []
    for c in comps:
        ct = []
        for t in _tokens(c):
            if t.isdigit():
                t = t.lstrip("0") or "0"
            t = ORDINAL_WORDS.get(t, t)
            t = ADDR_ABBR.get(t, t)
            ct.append(t)
        # a component that is only a state code / state name -> canonical state
        key = " ".join(ct)
        if len(ct) <= 3 and key in STATES:
            ct = STATES[key].split()
        comp_toks.append(ct)
        toks.extend(ct)
    nums = [t for t in toks if t.isdigit() and t != postcode.lstrip("0")]
    core = [t for t in toks if t not in ADDR_DROP]
    words = [t for t in core if not t.isdigit()]
    city = ""
    # heuristic: the last component that is not a pure number / state name
    states_full = _STATE_NAMES
    for ct in reversed(comp_toks):
        w = [t for t in ct if not t.isdigit()]
        if w and " ".join(ct) not in states_full and not any(ch.isdigit() for ch in "".join(ct)):
            city = " ".join(w[-2:])
            break
    return dict(
        addr_norm=" ".join(toks),
        addr_core=" ".join(core),
        addr_words=" ".join(words),
        postcode=postcode,
        house_nums=" ".join(dict.fromkeys(nums)),
        primary_num=nums[0] if nums else "",
        landmark=landmark,
        city_guess=city,
        addr_comps="|".join(" ".join(c) for c in comp_toks),
        addr_empty=int(not toks),
    )


_STATE_NAMES = set(STATES.values())


def _norm_chunk(args):
    names, addrs = args
    rows = []
    for n, a in zip(names, addrs):
        d = norm_name(n)
        d.update(norm_addr(a))
        rows.append(d)
    return pd.DataFrame(rows)


def normalize_df(df: pd.DataFrame, n_jobs: int = None, chunk: int = 50_000) -> pd.DataFrame:
    """Return a DataFrame of normalised fields aligned with df's rows."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 2)
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(n_jobs) as p:
        parts = p.map(_norm_chunk, jobs)
    out = pd.concat(parts, ignore_index=True)
    for c in ("entity_id", "country", "source"):
        out[c] = df[c].values
    return out
