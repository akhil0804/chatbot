import os
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
PORT = int(os.getenv("MYSQL_PORT", "3306"))
USER = os.getenv("MYSQL_USER")
PASSWORD = os.getenv("MYSQL_PASSWORD")
DATABASE = os.getenv("MYSQL_DB")

def get_conn():
    return pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE,
        autocommit=True, charset="utf8mb4", cursorclass=DictCursor
    )

def query(sql: str, params: tuple = ()):
    if not sql.strip().lower().startswith("select") or ";" in sql:
        raise ValueError("Only single SELECTs allowed.")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()

# def query(sql, params=None):
#     conn = get_conn()
#     try:

#         with conn.cursor() as cur:
#             cur.execute(sql, params or ())
#             cols = [c[0] for c in cur.description]
#             rows = [dict(zip(cols, r)) for r in cur.fetchall()]
#             return rows
#     finally:
#         conn.close()
    