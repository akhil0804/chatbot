# tools/metadata.py
from typing import Dict, List
from db import get_conn

def load_schema(include_views: bool = True) -> Dict[str, List[str]]:
    sql = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    schema: Dict[str, List[str]] = {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                t = row["TABLE_NAME"]
                c = row["COLUMN_NAME"]
                schema.setdefault(t, []).append(c)
    finally:
        conn.close()
    return schema