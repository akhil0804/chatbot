import yaml
from typing import Dict, Any, List, Tuple
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def load_catalog(path: str = "config/semantic_catalog.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
            return doc or {}
    except Exception:
        return {}

def build_context_and_allowed(catalog: Dict[str, Any], question: str, max_cols=18) -> Tuple[str, Dict[str, List[str]]]:
    ents = catalog.get("entities", {})
    # rank (very simple): prefer entities mentioned in question
    ql = question.lower()
    ranked = sorted(
        ents.items(),
        key=lambda kv: (kv[0].lower() in ql or (kv[1].get("label","").lower() in ql)),
        reverse=True
    )
    chosen = ranked[:2] if ranked else list(ents.items())[:2]

    lines: List[str] = []
    allowed: Dict[str, List[str]] = {}
    for table, meta in chosen:
        cols_map = (meta.get("columns") or {})
        cols = list(cols_map.keys())
        allowed[table] = cols
        lines.append(f"- Table `{table}` ({meta.get('label','')}): {meta.get('description','')}".strip())
        shown = 0
        for col in cols:
            if shown >= max_cols:
                lines.append("  • …")
                break
            spec = cols_map.get(col) or {}
            if isinstance(spec, dict):
                desc = spec.get("desc","")
                role = spec.get("role")
                vals = spec.get("values")
                if role == "enum" and vals:
                    desc = f"{desc} [enum: {', '.join(vals[:6])}{'…' if len(vals)>6 else ''}]"
                elif role:
                    desc = f"{desc} [{role}]"
            else:
                desc = str(spec)
            lines.append(f"  • `{col}` — {desc}")
            shown += 1

    # relationships
    rels = catalog.get("relationships") or []
    if rels:
        lines.append("Relationships:")
        for r in rels:
            lines.append(f"  • `{r['from']}` → `{r['to']}` ({r.get('type','')}) — {r.get('desc','')}")

    # global roles
    roles = (catalog.get("global") or {}).get("roles") or {}
    if roles:
        lines.append("Global roles:")
        for role, cols in roles.items():
            lines.append(f"  • {role}: {', '.join(cols)}")

    return "\n".join(lines), allowed

def enum_values(catalog: Dict[str, Any], enum_name: str) -> Tuple[List[str], Dict[str, Any]]:
    enums = (catalog.get("entities") or {})
    # flatten enums by column name across entities
    canonical: List[str] = []
    synonyms: Dict[str, Any] = {}
    for _, meta in enums.items():
        colspec = (meta.get("columns") or {}).get(enum_name)
        if isinstance(colspec, dict) and colspec.get("role") == "enum":
            canonical = colspec.get("values") or canonical
            synonyms = colspec.get("synonyms") or synonyms
    return canonical, synonyms