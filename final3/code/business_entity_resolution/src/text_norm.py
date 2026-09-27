"""Text normalisation for business names and addresses.

Design goals
  * country-agnostic: nothing here depends on the country label, so unseen countries
    (France in the test set) go through exactly the same code path;
  * cross-script: Indic-script tokens are mapped to Latin through a dictionary learned
    from the training pairs (see `build_dicts.py`), with `anyascii` as fallback;
  * canonical short forms: every abbreviation variant is mapped to ONE short token
    ("street"/"st"/"str" -> "st", "private"/"pvt" -> "pvt", "rue"/"r" -> "rue", ...)
    so that exact token comparison works across sources.
"""
import re
import unicodedata

from anyascii import anyascii

# ------------------------------------------------------------------ canonical maps
_ADDR_CANON = {
    # generic / US street types
    "street": "st", "str": "st", "st": "st", "saint": "st", "sainte": "ste",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "place": "pl", "pl": "pl", "plc": "pl",
    "square": "sq", "sq": "sq",
    "circle": "cir", "cir": "cir", "circ": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "trail": "trl", "trl": "trl",
    "terrace": "terr", "terr": "terr",
    "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg", "bld": "bldg",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "north": "n", "south": "s", "so": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    "post": "po", "box": "box",
    # India
    "near": "near", "nr": "near", "opposite": "opp", "opp": "opp",
    "district": "dist", "dist": "dist", "distt": "dist",
    "colony": "colony", "col": "colony",
    "nagar": "nagar", "ngr": "nagar",
    "sector": "sector", "sec": "sector",
    "phase": "phase", "ph": "phase",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "ground": "grd", "grd": "grd",
    # France
    "rue": "rue", "r": "rue",
    "impasse": "imp", "imp": "imp",
    "chemin": "chem", "chem": "chem", "ch": "chem",
    "route": "rte", "rte": "rte",
    "allee": "allee", "all": "allee",
    "quai": "quai", "qu": "quai",
    "faubourg": "fg", "fbg": "fg", "fg": "fg",
    "cours": "crs", "crs": "crs",
    "residence": "res", "res": "res",
    "batiment": "bat", "bat": "bat",
    "etage": "etg", "etg": "etg",
}
# tokens that only mark "a number follows" -> dropped
_ADDR_DROP = {"no", "nos", "num", "number", "h", "hno", "door", "plot", "and"}

_NAME_CANON = {
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co", "comp": "co",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "lt": "ltd",
    "private": "pvt", "pvt": "pvt", "prv": "pvt", "pte": "pvt", "priv": "pvt",
    "public": "pub", "pub": "pub",
    "and": "and", "et": "and", "n": "and",
    "brothers": "bros", "bros": "bros", "bro": "bros",
    "international": "intl", "intl": "intl", "intnl": "intl",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc",
    "industries": "ind", "industry": "ind", "inds": "ind",
    "technologies": "tech", "technology": "tech", "tech": "tech",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent",
    "associates": "assoc", "association": "assoc", "assoc": "assoc",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "saint": "st", "st": "st",
    "the": "the", "dba": "dba",
}
LEGAL_TOKENS = {
    "inc", "corp", "co", "ltd", "pvt", "pub", "llc", "llp", "lp", "plc", "pllc", "pc", "lllp",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "ei", "scp", "selarl", "gie", "scop",
    "gmbh", "ag", "bv", "nv", "pty", "opc",
}
STOP_TOKENS = {"the", "and", "of", "a", "an", "de", "des", "du", "la", "le", "les", "d", "l", "en", "au", "aux"}
NULL_TOKENS = {"null", "none", "nan", "na", "n/a", "<null>", "nil", "unknown"}

_TLD = r"(?:com|net|org|biz|info|co|in|fr|us|io|co\.in|org\.in|net\.in)"
_DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\." + _TLD + r"\b", re.I)
_DBA_RE = re.compile(r"\s+(?:doing business as|d/b/a|dba|d\.b\.a\.|a/k/a|aka|t/a|trading as)\s+|\s*\|\s*", re.I)
_INITIALS_RE = re.compile(r"\b(?:[^\W\d_]\.){2,}")
_ELISION_RE = re.compile(r"\b([dljmnst])['’]", re.I)
_APOS_RE = re.compile(r"['’‘`´]")
_BIS_RE = re.compile(r"(\d)(bis|ter)\b", re.I)
_SPLIT_RE = re.compile(r"[\s,.;:()\[\]{}<>|/\\_#*\"“”«»!?=~^%$@°º—–\-]+")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIGITS_RE = re.compile(r"\d+")


def is_native(tok: str) -> bool:
    """True if the token contains a letter outside the Latin script blocks."""
    for c in tok:
        o = ord(c)
        if o > 0x24F and not (0x1E00 <= o <= 0x1EFF) and unicodedata.category(c)[0] in "LM":
            return True
    return False


def has_native(s) -> bool:
    return bool(s) and any(is_native(t) for t in s.split())


def comp_key(comp: str) -> str:
    """Key used for address-component synonym lookup (keeps native script and its marks)."""
    c = unicodedata.normalize("NFKC", comp).lower()
    c = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in c)
    return " ".join(c.split())


def raw_tokens(s: str) -> list:
    """Split a raw string into tokens without transliteration or canonicalisation."""
    if not s:
        return []
    s = unicodedata.normalize("NFKC", s)
    s = _ELISION_RE.sub(r"\1 ", s)
    s = _APOS_RE.sub("", s)
    s = _INITIALS_RE.sub(lambda m: m.group(0).replace(".", "") + " ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = _BIS_RE.sub(r"\1 \2", s)
    return [t for t in _SPLIT_RE.split(s) if t]


def latin_token(tok: str) -> str:
    """Basic Latin form of one token (accents removed, lowercase)."""
    return _NONALNUM_RE.sub("", anyascii(tok).lower())


class Normalizer:
    """Stateful normaliser holding the learned dictionaries.

    token_dict : native-script token -> Latin token   (learned from train pairs)
    comp_dict  : (country, component key) -> canonical component (learned, e.g.
                 ("US", "north carolina") -> "nc", ("India", "calcutta") -> "kolkata")
    """

    def __init__(self, token_dict=None, comp_dict=None):
        self.token_dict = token_dict or {}
        self.comp_dict = comp_dict or {}

    # ---------------------------------------------------------------- tokens
    def _to_latin(self, tok):
        if is_native(tok):
            mapped = self.token_dict.get(tok)
            if mapped is not None:
                return [mapped]
        return [t for t in _NONALNUM_RE.split(anyascii(tok).lower()) if t]

    def _latin_tokens(self, s):
        out = []
        for t in raw_tokens(s):
            for lt in self._to_latin(t):
                if lt.isdigit():
                    lt = lt.lstrip("0") or "0"
                out.append(lt)
        return out

    # ---------------------------------------------------------------- name
    def name(self, raw):
        """Return dict with normalised name representations."""
        raw = raw or ""
        low = raw.lower()
        is_domain = bool(_DOMAIN_RE.search(low))
        text = _DOMAIN_RE.sub(lambda m: " " + m.group(1) + " ", raw)
        parts = [p for p in _DBA_RE.split(text) if p and p.strip()]
        has_dba = len(parts) > 1
        toks = [_NAME_CANON.get(t, t) for t in self._latin_tokens(" ".join(parts))]
        toks = [t for t in toks if t not in ("www", "dba", "http", "https")]
        core = [t for t in toks if t not in LEGAL_TOKENS and t not in STOP_TOKENS]
        # alternative name = the longest DBA part (trade name vs legal name)
        alt = ""
        if has_dba:
            alts = []
            for p in parts:
                pt = [_NAME_CANON.get(t, t) for t in self._latin_tokens(p)]
                pt = [t for t in pt if t not in LEGAL_TOKENS and t not in STOP_TOKENS and t != "www"]
                if pt:
                    alts.append(" ".join(pt))
            alt = "|".join(alts)
        return {
            "name": " ".join(toks),
            "name_core": " ".join(core) if core else " ".join(toks),
            "name_alt": alt,
            "name_is_domain": is_domain,
            "name_has_dba": has_dba,
        }

    # ---------------------------------------------------------------- address
    def address(self, raw, country):
        raw = raw or ""
        comps = []
        for c in raw.split(","):
            k = comp_key(c)
            if not k or k in NULL_TOKENS:
                continue
            mapped = self.comp_dict.get((country, k))
            comps.append(mapped if mapped is not None else c)
        toks = []
        for c in comps:
            for t in self._latin_tokens(c):
                if t in NULL_TOKENS or t in _ADDR_DROP:
                    continue
                toks.append(_ADDR_CANON.get(t, t))
        nums = sorted({d.lstrip("0") or "0" for d in _DIGITS_RE.findall(" ".join(toks))})
        return {"addr": " ".join(toks), "addr_nums": " ".join(nums)}
