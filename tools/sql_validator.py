# tools/sql_validator.py
import re
from typing import Dict, List, Tuple
COMMENT_LINE = re.compile(r"--.*?$", re.M)
COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)

def _strip_comments(s: str) -> str:
    s = COMMENT_BLOCK.sub("", s)
    s = COMMENT_LINE.sub("", s)
    return s

SAFE_START = re.compile(r"^\s*(select)\b", re.I)
FORBIDDEN  = re.compile(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke)\b", re.I)
PH_Q       = re.compile(r"\?")
PH_PYMYSQL = re.compile(r"%s")

TABLE_BACK  = r"`([A-Za-z0-9 _]+)`"
STAR_ANY    = re.compile(r"\bselect\s+(.*?)\s+from\s+", re.I | re.S)
TABLE_STAR  = re.compile(TABLE_BACK + r"\.\*", re.I)

PLAIN_STAR  = re.compile(r"\bselect\s+\*\s+from\b", re.I)

# Extract simple table aliases like: FROM `tabSales Order` AS so  /  JOIN `tabSales Order` so
ALIAS_FROM = re.compile(r"\bFROM\s+`([A-Za-z0-9 _]+)`\s+(?:AS\s+)?`?([A-Za-z0-9_]+)`?\b", re.I)
ALIAS_JOIN = re.compile(r"\bJOIN\s+`([A-Za-z0-9 _]+)`\s+(?:AS\s+)?`?([A-Za-z0-9_]+)`?\b", re.I)

def _extract_aliases(sql: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for tbl, alias in ALIAS_FROM.findall(sql):
        if alias.lower() not in {"on", "using", "where", "group", "order", "limit", "left", "right", "inner", "outer", "join"}:
            aliases[alias] = tbl
    for tbl, alias in ALIAS_JOIN.findall(sql):
        if alias.lower() not in {"on", "using", "where", "group", "order", "limit", "left", "right", "inner", "outer", "join"}:
            aliases[alias] = tbl
    return aliases

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:sql|mysql)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _strip_trailing_semicolon(s: str) -> str:
    return re.sub(r";\s*$", "", s)

def to_pymysql_placeholders(s: str) -> str:
    return s if PH_PYMYSQL.search(s) else PH_Q.sub("%s", s)

def count_placeholders_any(s: str) -> int:
    return len(PH_Q.findall(s)) + len(PH_PYMYSQL.findall(s))

def _quote_table_names(s: str, schema: Dict[str, List[str]]) -> str:
    for table in sorted(schema.keys(), key=len, reverse=True):
        patt = re.compile(rf"(?i)(\bfrom\s+|\bjoin\s+){re.escape(table)}\b")
        s = patt.sub(lambda m: f"{m.group(1)}`{table}`", s)
    return s

def _quote_qualified_columns(s: str, schema: Dict[str, List[str]]) -> str:
    table_alt = "|".join(re.escape(t) for t in sorted(schema.keys(), key=len, reverse=True))
    patt = re.compile(rf"(?<!`)\b({table_alt})\b\.(?!`)([A-Za-z0-9_]+)\b")
    return patt.sub(lambda m: f"`{m.group(1)}`.`{m.group(2)}`", s)

def _all_cols_for(table: str, schema: Dict[str, List[str]]) -> str:
    cols = schema.get(table, [])
    return ", ".join(f"`{table}`.`{c}`" for c in cols)

def _expand_select_star(sql: str, schema: Dict[str, List[str]]) -> str:
    s = sql
    s = TABLE_STAR.sub(lambda m: _all_cols_for(m.group(1), schema), s)
    if PLAIN_STAR.search(s):
        m = re.search(r"\bfrom\s+`([A-Za-z0-9 _]+)`\b", s, re.I)
        if m:
            table = m.group(1)
            cols = _all_cols_for(table, schema)
            s = STAR_ANY.sub(f"SELECT {cols} FROM ", s)
    return s

# light whitelist: ensure selected columns exist (after star expansion)
# This now supports backticked aliases like `so`.`name` by mapping aliases → real tables.
COL_TOKEN = re.compile(r"`([A-Za-z0-9 _]+)`\.`([A-Za-z0-9_]+)`", re.I)

def _columns_in_schema(sql: str, schema: Dict[str, List[str]]) -> List[str]:
    unknown: List[str] = []
    alias_map = _extract_aliases(sql)
    for t, c in COL_TOKEN.findall(sql):
        canonical = t
        if t not in schema and t in alias_map:
            canonical = alias_map[t]
        if canonical not in schema or c not in schema.get(canonical, []):
            unknown.append(f"{t}.{c}")
    return unknown

def validate_sql(sql: str, schema: Dict[str, List[str]]) -> str:
    s = _strip_code_fences(sql)
    s = _strip_comments(s)
    if not sql:
        raise ValueError("Empty SQL from LLM")
    s = _strip_code_fences(sql)
    if s.upper() == "NO_DB_NEEDED":
        return "NO_DB_NEEDED"
    s = _strip_trailing_semicolon(s)
    if ";" in s: raise ValueError("Multiple statements not allowed.")
    if not SAFE_START.match(s): raise ValueError("Only SELECT is allowed.")
    if FORBIDDEN.search(s): raise ValueError("Forbidden statement detected.")

    # quote tables/columns so spaces are safe
    s = _quote_table_names(s, schema)
    s = _quote_qualified_columns(s, schema)

    # expand stars now that tables/columns are quoted
    s = _expand_select_star(s, schema)

    # enforce columns exist
    unknown = _columns_in_schema(s, schema)
    if unknown:
        raise KeyError("Unknown columns: " + ", ".join(unknown))

    # if re.search(r"\blimit\s+\d+\b", s, re.I) is None:
    #     s = f"{s} LIMIT 50"

    return to_pymysql_placeholders(s)