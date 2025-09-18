import os, re, json, logging, time
from typing import Dict, Any, List, Set
from .base import Agent
from llm.base import LLM
from tools.sql_validator import validate_sql, count_placeholders_any
from tools.semantic_catalog import load_catalog, build_context_and_allowed
from tools.metadata import load_schema  # optional: live DB columns
from db import query

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

PARAMS_LINE = re.compile(r"^\s*--PARAMS\s*:\s*(\[.*\])\s*$", re.I | re.M)

# --- logging setup ---
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class NL2SQLAgent(Agent):
    # Safe fallback column lists for key tables when live schema or catalog narrowing misses them
    SAFE_FALLBACK_COLS: Dict[str, List[str]] = {
        # --- Order to Cash (existing + verified) ---
        "tabSales Order": [
            "name", "customer_name", "status", "transaction_date", "delivery_date", "total", "currency"
        ],
        "tabSales Order Item": [
            "name", "parent", "item_code", "item_name", "qty", "uom", "rate", "amount", "description",
            "prevdoc_docname", "quotation_item"
        ],
        "tabQuotation": [
            "name", "opportunity", "customer_name", "transaction_date", "valid_till", "currency"
        ],
        "tabQuotation Item": [
            "parent", "item_code", "qty", "rate", "description", "item_name", "unit_price", "margin",
            "supplier_description"
        ],
        "tabDelivery Note": [
            "name", "customer_name", "posting_date", "status"
        ],
        "tabDelivery Note Item": [
            "name", "parent", "item_code", "item_name", "qty", "uom", "description",
            "against_sales_order", "so_detail"
        ],
        "tabSales Invoice": [
            "name", "customer", "posting_date", "due_date", "total", "currency", "status"
        ],
        "tabSales Invoice Item": [
            "name", "parent", "item_code", "item_name", "qty", "uom", "rate", "amount", "description",
            "so_detail", "dn_detail", "delivery_note"
        ],

        # --- Opportunity → RFQ → Supplier Quotation (added from semantic catalog) ---
        "tabOpportunity": [
            "name", "customer_name", "status", "customer_reference_no", "vessel", "sales_person",
            "customer_department", "port_of_delivery", "priority", "transaction_date", "purchaser",
            "assigned_to_purchaser_on", "contact_person", "contact_email", "contact_mobile",
            "warehouse", "currency", "etd"
        ],
        "tabOpportunity Item": [
            "name", "parent", "item_code", "item_name", "stock_availability_status", "stock_qty",
            "base_price", "supplier_list", "description", "qty", "uom"
        ],
        "tabRequest for Quotation": [
            "name", "opportunity", "transaction_date"
        ],
        "tabRequest for Quotation Item": [
            "parent", "item_code", "qty"
        ],
        "tabRequest for Quotation Supplier": [
            "parent", "supplier", "contact", "email_id", "send_email", "email_sent"
        ],
        "tabSupplier Quotation": [
            "name", "opportunity", "supplier", "transaction_date", "currency"
        ],
        "tabSupplier Quotation Item": [
            "parent", "item_code", "rate", "supplier_description"
        ],

        # --- Masters referenced in the flow (added) ---
        "Supplier": [
            "name", "email_id", "contact_mobile", "status"
        ],
        "Customer": [
            "name", "email_id", "contact_mobile", "priority", "status"
        ],
        "Item": [
            "item_code", "item_name", "uom", "description"
        ],

        # --- Purchase flow (existing + verified) ---
        "tabPurchase Order": [
            "name", "supplier", "transaction_date", "schedule_date", "status", "currency", "total"
        ],
        "tabPurchase Order Item": [
            "name", "parent", "item_code", "item_name", "qty", "uom", "rate", "amount",
            "sales_order", "sales_order_item", "customer_enquiry"
        ],
        "tabPurchase Receipt": [
            "name", "supplier", "posting_date", "status"
        ],
        "tabPurchase Receipt Item": [
            "name", "parent", "item_code", "item_name", "received_qty", "qty", "rejected_qty", "uom",
            "rate", "amount", "purchase_order", "sales_order", "sales_order_item", "purchase_order_item"
        ],
    }
    def _parse_alias_map(self, sql: str) -> Dict[str, str]:
        """Build alias -> table mapping from FROM/JOIN clauses."""
        alias_map: Dict[str, str] = {}
        try:
            # FROM `table` [AS] alias
            for tbl, alias in re.findall(r"FROM\s+`([^`]+)`\s+(?:AS\s+)?([a-zA-Z_]\w*)", sql, flags=re.I):
                alias_map[alias] = tbl
            # JOIN `table` [AS] alias
            for tbl, alias in re.findall(r"JOIN\s+`([^`]+)`\s+(?:AS\s+)?([a-zA-Z_]\w*)", sql, flags=re.I):
                alias_map[alias] = tbl
        except Exception:
            pass
        return alias_map

    def _strict_alias_column_check_and_autocorrect(self, sql: str, schema_for_val: Dict[str, List[str]]):
        """
        If an alias-qualified column (alias.`col` / alias."col" / alias.col) doesn't exist on that alias's
        table but exists on exactly one other joined table, rewrite the alias. Return (new_sql, fixes).
        """
        alias_map = self._parse_alias_map(sql)
        fixes: List[str] = []
        if not alias_map:
            return sql, fixes

        schema_cols = {t: set(cols) for t, cols in (schema_for_val or {}).items()}

        def correct_for_matches(pattern_find, build_pattern, build_repl, label: str):
            nonlocal sql, fixes
            try:
                matches = pattern_find(sql)
            except re.error:
                matches = []
            # dedupe pairs
            seen = set(matches)
            for alias, col in seen:
                table = alias_map.get(alias)
                if not table:
                    continue
                # OK if column belongs to referenced table
                if col in schema_cols.get(table, set()):
                    continue
                # Otherwise, find if exactly one other alias has this column
                candidate_aliases = [
                    other_alias
                    for other_alias, other_table in alias_map.items()
                    if other_alias != alias and col in schema_cols.get(other_table, set())
                ]
                if len(candidate_aliases) == 1:
                    new_alias = candidate_aliases[0]
                    pat = build_pattern(alias, col)
                    sql_new, nsubs = pat.subn(build_repl(new_alias, col), sql)
                    if nsubs > 0 and sql_new != sql:
                        fixes.append(f"Rewrote {alias}.{col} -> {new_alias}.{col} ({label})")
                        sql = sql_new

        # Backticked alias.`col`
        correct_for_matches(
            lambda s: re.findall(r"([A-Za-z_]\w*)\s*\.\s*`([^`]+)`", s),
            lambda a, c: re.compile(rf"\b{re.escape(a)}\s*\.\s*`{re.escape(c)}`"),
            lambda na, c: f"{na}.`{c}`",
            "backtick"
        )
        # Double-quoted alias."col"
        correct_for_matches(
            lambda s: re.findall(r'([A-Za-z_]\w*)\s*\.\s*"([^"]+)"', s),
            lambda a, c: re.compile(rf'\b{re.escape(a)}\s*\.\s*"{re.escape(c)}"'),
            lambda na, c: f'{na}."{c}"',
            "double-quote"
        )
        # Bare alias.col
        correct_for_matches(
            lambda s: re.findall(r"([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)", s),
            lambda a, c: re.compile(rf"\b{re.escape(a)}\s*\.\s*{re.escape(c)}\b"),
            lambda na, c: f"{na}.{c}",
            "bare"
        )

        return sql, fixes
    
    def _ensure_table_in_schema(self, combined: Dict[str, List[str]], table: str) -> None:
        """Ensure `table` exists in `combined` with either live schema cols or safe fallback cols."""
        if table in combined:
            return
        live_cols = self.schema.get(table, []) if hasattr(self, "schema") and isinstance(self.schema, dict) else []
        if live_cols:
            combined[table] = list(live_cols)
            return
        fb = self.SAFE_FALLBACK_COLS.get(table)
        if fb:
            combined[table] = list(fb)

    def _extract_tables_from_sql(self, sql: str) -> Set[str]:
        """Best-effort parse of table names used in FROM/JOIN clauses."""
        tables: Set[str] = set()
        try:
            # FROM `table`
            for t in re.findall(r"FROM\s+`([^`]+)`", sql, flags=re.I):
                tables.add(t)
            # JOIN `table`
            for t in re.findall(r"JOIN\s+`([^`]+)`", sql, flags=re.I):
                tables.add(t)
        except Exception:
            pass
        return tables

    def _widen_schema_for_sql(self, sql: str, schema_for_val: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Add any referenced tables that are missing from the narrowed schema using live or fallback cols."""
        widened = dict(schema_for_val)
        for t in self._extract_tables_from_sql(sql):
            if t not in widened:
                self._ensure_table_in_schema(widened, t)
        return widened
    def __init__(self, llm: LLM,
                 catalog_path="config/semantic_catalog.yaml",
                 nl2sql_prompt_path="config/prompts/nl2sql.txt",
                 analyst_prompt_path="config/prompts/answer_analyst.txt"):
        self.llm = llm
        self.catalog = load_catalog(catalog_path)
        # load live DB schema to cross-check/expand; falls back to db_schema.yaml if you prefer
        # self.schema = load_schema()
        self.schema = {}

        with open(nl2sql_prompt_path, "r") as f:
            self.template = f.read()
        with open(analyst_prompt_path, "r") as f:
            self.analyst_template = f.read()

        LOGGER.info("NL2SQLAgent initialized; DEBUG=%s", DEBUG)


    def _needs_job_order_context(self, text: str) -> bool:
        q = (text or "").lower()
        # cover common synonyms/phrases
        return any(k in q for k in [
            "job order", " jo ", "jo ", " jo", "sales order", "delivery note",
            "delayed", "delay", "overdue", "late"
        ])

    def _needs_quotation_success_context(self, text: str) -> bool:
        q = (text or "").lower()
        return any(k in q for k in [
            "success rate", "conversion rate", "win rate", "quoted vs", "quoted versus", "quotation success"
        ]) and ("quotation" in q or "quote" in q)

    def _needs_opportunity_flow_context(self, text: str) -> bool:
        q = (text or "").lower()
        return any(k in q for k in [
            "opportunity", "rfq", "request for quotation", "invited supplier", "supplier quotation",
            "quoted supplier", "unquoted supplier", "supplier responded", "supplier response"
        ])

    def _heuristic_sql_quotation_success(self) -> str:
        # Success rate of customer quotations converting to sales orders (via Sales Order Item)
        return (
            "SELECT\n"
            "  COUNT(DISTINCT q.`name`) AS `quoted_count`,\n"
            "  COUNT(DISTINCT so.`name`) AS `confirmed_count`,\n"
            "  ROUND(\n"
            "    COUNT(DISTINCT so.`name`) / NULLIF(COUNT(DISTINCT q.`name`), 0) * 100, 2\n"
            "  ) AS `success_rate_pct`\n"
            "FROM `tabQuotation` AS q\n"
            "LEFT JOIN `tabSales Order Item` AS soi\n"
            "  ON soi.`prevdoc_docname` = q.`name`\n"
            "LEFT JOIN `tabSales Order` AS so\n"
            "  ON so.`name` = soi.`parent`\n"
            " AND so.`status` IN ('Submitted','To Deliver','To Bill','Completed')\n"
            "LIMIT 500"
        )

    def _heuristic_sql_job_order_delays(self) -> str:
        # Fallback SQL for: Which job orders are delayed? Reason (supplier delay, approval pending, etc.)
        return (
            "SELECT\n"
            "  so.`name` AS `sales_order_id`,\n"
            "  so.`customer_name`,\n"
            "  so.`status`,\n"
            "  so.`transaction_date`,\n"
            "  so.`delivery_date`,\n"
            "  so.`total`,\n"
            "  so.`currency`,\n"
            "  CASE\n"
            "    WHEN so.`status` = 'Draft' AND so.`delivery_date` < CURRENT_DATE THEN 'Approval pending'\n"
            "    WHEN SUM(CASE WHEN po.`schedule_date` < CURRENT_DATE AND COALESCE(prsum.`received_qty`,0) < poi.`qty` THEN 1 ELSE 0 END) > 0 THEN 'Supplier delay'\n"
            "    WHEN so.`delivery_date` < CURRENT_DATE THEN 'Internal delay'\n"
            "    ELSE NULL\n"
            "  END AS `delay_reason`\n"
            "FROM `tabSales Order` AS so\n"
            "LEFT JOIN `tabSales Order Item` AS soi\n"
            "  ON soi.`parent` = so.`name`\n"
            "LEFT JOIN `tabPurchase Order Item` AS poi\n"
            "  ON poi.`sales_order_item` = soi.`name` OR poi.`sales_order` = so.`name`\n"
            "LEFT JOIN `tabPurchase Order` AS po\n"
            "  ON po.`name` = poi.`parent`\n"
            "LEFT JOIN (\n"
            "  SELECT `purchase_order_item`, SUM(`qty`) AS `received_qty`\n"
            "  FROM `tabPurchase Receipt Item`\n"
            "  GROUP BY `purchase_order_item`\n"
            ") AS prsum\n"
            "  ON prsum.`purchase_order_item` = poi.`name`\n"
            "WHERE so.`status` NOT IN ('Completed','Closed','Cancelled')\n"
            "GROUP BY so.`name`, so.`customer_name`, so.`status`, so.`transaction_date`, so.`delivery_date`, so.`total`, so.`currency`\n"
            "HAVING\n"
            "  (so.`status` = 'Draft' AND so.`delivery_date` < CURRENT_DATE)\n"
            "  OR (SUM(CASE WHEN po.`schedule_date` < CURRENT_DATE AND COALESCE(prsum.`received_qty`,0) < poi.`qty` THEN 1 ELSE 0 END) > 0)\n"
            "  OR (so.`delivery_date` < CURRENT_DATE)\n"
            "ORDER BY so.`delivery_date` ASC\n"
            "LIMIT 500"
        )

    def _heuristic_repair(self, user_query: str, llm_sql: str, schema_for_val: Dict[str, List[str]], error_msg: str) -> str | None:
        """Simple fallback heuristics to produce a safe SELECT when LLM SQL fails validation/DB.
        Do not infer params; keep it SELECT-only with LIMIT 500.
        """
        LOGGER.debug("[repair] invoked; error_msg=%s", error_msg)
        uq = (user_query or "").lower()
        if self._needs_job_order_context(uq) and ("delay" in uq or "delayed" in uq or "overdue" in uq or "late" in uq):
            LOGGER.info("[repair] using job_order_delays heuristic")
            return self._heuristic_sql_job_order_delays()
        if self._needs_quotation_success_context(uq):
            LOGGER.info("[repair] using quotation_success heuristic")
            return self._heuristic_sql_quotation_success()
        return None

    def _sanitize_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return rows[:50]


    def _rows_to_json(self, rows: List[Dict[str, Any]], cap: int = 50) -> str:
        try:
            # Compact but consistent JSON for the analyst prompt
            return json.dumps(rows[:cap], ensure_ascii=False)
        except Exception:
            # Fallback: key=value lines (still parseable-ish by LLM)
            lines = []
            sample = rows[:cap]
            for r in sample:
                parts = [f"{k}={r.get(k)}" for k in r.keys()]
                lines.append("; ".join(parts))
            return "\n".join(lines)

    def _analyze(self, question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
        """Use the analyst prompt to generate a crisp answer from result rows.
        - No disclaimers, no instructions to the user, no speculative fields.
        - Keep it grounded strictly to the provided rows and question.
        - Prefer concise bullet points or a one-liner when the user asks for a single row.
        """
        if not rows:
            return "No matching records were found."

        # Detect if the user asked for a single row / top 1
        ql = (question or "").lower()
        want_one = any(k in ql for k in ["only one row", "top 1", "top1", "first row", "single row", "one row"]) or re.search(r"\blimit\s*1\b", ql)

        rows_json = self._rows_to_json(rows, cap=1 if want_one else 50)

        # Build an analysis prompt from template or fallback
        if isinstance(getattr(self, "analyst_template", None), str) and self.analyst_template.strip():
            prompt = (self.analyst_template
                      .replace("{{question}}", question)
                      .replace("{{row_count}}", str(len(rows)))
                      .replace("{{rows_json}}", rows_json)
                      .replace("{{sql}}", sql))
        else:
            prompt = (
                "You are a precise data analyst. Answer succinctly and factually from the data given.\n"
                "- Do NOT add disclaimers or suggest checking filters.\n"
                "- If the user wants one row, answer with that single row.\n"
                "- If aggregated fields like cnt/count exist, summarize the top insights.\n"
                "- Keep the answer under 8 lines unless absolutely necessary.\n\n"
                f"Question: {question}\n"
                f"Row count: {len(rows)}\n"
                f"Rows(JSON): {rows_json}\n"
            )
        try:
            return self.llm.complete(prompt, system="You are a helpful data analyst.", temperature=0)
        except Exception as e:
            LOGGER.warning("[analyst] LLM failed, falling back to deterministic summarizer: %s", e)
            return self._summarize(question, rows)


    def _summarize(self, question: str, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "No matching records were found."
        sample = rows[:50]
        keys = list(sample[0].keys())
        lines = [f"Total rows: {len(rows)}"]
        for i, r in enumerate(sample, 1):
            parts = [f"{k}: {r.get(k)}" for k in keys if r.get(k) not in (None, "")]
            lines.append(f"{i}. " + ", ".join(parts))
        if len(rows) > 50:
            lines.append("… (truncated) …")
        return "\n".join(lines)

    def _render(self, question: str) -> str:
        sem_ctx, allowed_from_catalog = build_context_and_allowed(self.catalog, question)
        self.allowed_from_catalog = allowed_from_catalog  # stash for validation narrowing
        self._current_question = question
        LOGGER.debug("[render] question=%r sem_ctx_len=%d", (question[:120] + ('…' if len(question) > 120 else '')), len(sem_ctx))
        return (self.template
                .replace("{{question}}", question)
                .replace("{{semantic_context}}", sem_ctx))

    def _split_sql_and_params(self, text: str) -> tuple[str, List[Any]]:
        params: List[Any] = []
        m = PARAMS_LINE.search(text)
        if m:
            try:
                params = json.loads(m.group(1))
            except Exception:
                params = []
            sql = PARAMS_LINE.sub("", text).strip()
        else:
            sql = text.strip()
        LOGGER.debug("[split] params_count=%d", len(params))
        return sql, params

    def _narrow_schema_by_catalog(self) -> Dict[str, List[str]]:
        """Build a validation schema allowing union fallback with catalog and safe fallbacks.
        - Prefer intersection when both live schema and catalog list columns for a table.
        - If live schema is missing or overlap is empty, fall back to catalog columns.
        - For job-order/quotation intents, forcibly ensure key header tables exist via live or SAFE_FALLBACK_COLS.
        """
        allowed = dict(self.allowed_from_catalog or {})

        # Combine with live schema; prefer overlap, else fallback to whichever is present
        combined: Dict[str, List[str]] = {}
        all_tables = set(self.schema.keys()) | set(allowed.keys()) if isinstance(self.schema, dict) else set(allowed.keys())
        for t in all_tables:
            live_cols = set(self.schema.get(t, [])) if isinstance(self.schema, dict) else set()
            cat_cols = set(allowed.get(t, []))
            if live_cols and cat_cols:
                cols = sorted(live_cols & cat_cols) or sorted(cat_cols)
            elif cat_cols:
                cols = sorted(cat_cols)
            else:
                cols = sorted(live_cols)
            if cols:
                combined[t] = cols

        # Heuristic reinforcement: ensure key tables are present with at least safe columns
        q = getattr(self, "_current_question", "") or ""
        if self._needs_job_order_context(q):
            for t in (
                "tabSales Order", "tabSales Order Item",
                "tabPurchase Order", "tabPurchase Order Item",
                "tabPurchase Receipt", "tabPurchase Receipt Item",
                "tabDelivery Note", "tabDelivery Note Item",
                "tabSales Invoice", "tabSales Invoice Item",
            ):
                self._ensure_table_in_schema(combined, t)
        if self._needs_quotation_success_context(q):
            for t in ("tabQuotation", "tabSales Order", "tabSales Order Item"):
                self._ensure_table_in_schema(combined, t)
        if self._needs_opportunity_flow_context(q):
            for t in (
                "tabOpportunity", "tabOpportunity Item",
                "tabRequest for Quotation", "tabRequest for Quotation Item", "tabRequest for Quotation Supplier",
                "tabSupplier Quotation", "tabSupplier Quotation Item",
                "Supplier", "Customer", "Item"
            ):
                self._ensure_table_in_schema(combined, t)

        try:
            tbls = list(combined.keys())
            preview = {t: len(combined[t]) for t in tbls[:12]}
            LOGGER.debug("[narrow] tables=%d preview=%s", len(tbls), preview)
        except Exception:
            pass
        return combined


    def run(self, user_query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        req_id = str(int(time.time() * 1000))
        LOGGER.info("[%s] question=%r", req_id, (user_query[:200] + ('…' if len(user_query) > 200 else '')))
        # 1) NL → SQL (+ explicit PARAMS line)
        prompt = self._render(user_query)
        llm_out = self.llm.complete(
            prompt,
            system="Translate to ONE safe MySQL SELECT then a --PARAMS: JSON array.",
            temperature=0
        )
        if DEBUG:
            print("[nl2sql] raw:", repr(llm_out))

        try:
            LOGGER.debug("[%s] llm_out_len=%d", req_id, len(llm_out))
        except Exception:
            pass

        sql_text, params = self._split_sql_and_params(llm_out)

        # 2) Validate/normalize SQL using schema ∩ catalog
        schema_for_val = self._narrow_schema_by_catalog()
        try:
            sql = validate_sql(sql_text, schema_for_val)
            sql, fixes = self._strict_alias_column_check_and_autocorrect(sql, schema_for_val)
            if fixes:
                LOGGER.info("[%s] autocorrected: %s", req_id, "; ".join(fixes))
        except Exception as e:
            LOGGER.warning("[%s] validation_failed(narrow); attempting widen-once: %s", req_id, e)
            widened = self._widen_schema_for_sql(sql_text, schema_for_val)
            try:
                sql = validate_sql(sql_text, widened)
                sql, fixes = self._strict_alias_column_check_and_autocorrect(sql, widened)
                if fixes:
                    LOGGER.info("[%s] autocorrected: %s", req_id, "; ".join(fixes))
                schema_for_val = widened  # carry widened schema forward
            except Exception as e2:
                LOGGER.exception("[%s] validation_failed(widen)", req_id)
                # Try a heuristic repair based on the question/context
                repaired = self._heuristic_repair(user_query, sql_text, schema_for_val, str(e2))
                if repaired:
                    try:
                        sql = validate_sql(repaired, schema_for_val)
                    except Exception:
                        return {"type": "nl2sql_invalid", "llm_sql": sql_text, "error": str(e2)}
                else:
                    return {"type": "nl2sql_invalid", "llm_sql": sql_text, "error": str(e2)}

        LOGGER.debug("[%s] validated_sql=%s", req_id, (sql[:500] + '…' if len(sql) > 500 else sql))

        if sql == "NO_DB_NEEDED":
            return {
                "type": "no_db_needed",
                "message": self.llm.complete(
                    f"Question: {user_query}\nReply briefly and helpfully without using a database.",
                    system="You are a helpful assistant.", temperature=0)
            }

        # 3) Params: use LLM-provided only; do not infer
        n_q = count_placeholders_any(sql)
        if n_q == 0:
            params = []
        elif len(params) != n_q:
            LOGGER.warning("[%s] need_parameters: required=%d provided=%d", req_id, n_q, len(params))
            return {
                "type": "need_parameters",
                "message": f"Query needs {n_q} parameter(s); found {len(params)}. Provide missing values.",
                "sql": sql,
                "llm_sql": sql_text
            }

        # 4) Execute
        try:
            LOGGER.info("[%s] sql query %s", req_id, sql)
            rows = query(sql, tuple(params))
            
            LOGGER.info("[%s] executed rows=%d", req_id, len(rows) if rows else 0)
        except Exception as e:
            LOGGER.exception("[%s] db_error: %s", req_id, e)
            # If DB rejects the SQL (e.g., wrong column/alias), try a heuristic repair and re-run
            repaired = self._heuristic_repair(user_query, sql_text, schema_for_val, str(e))
            if not repaired:
                return {"type": "db_error", "error": str(e), "sql": sql, "params": params}
            try:
                sql_r = validate_sql(repaired, schema_for_val)
                sql_r, fixes_r = self._strict_alias_column_check_and_autocorrect(sql_r, schema_for_val)
                if fixes_r:
                    LOGGER.info("[%s] autocorrected(repair): %s", req_id, "; ".join(fixes_r))
                # Recompute params for repaired SQL
                n_q_r = count_placeholders_any(sql_r)
                if n_q_r == 0:
                    params_r: List[Any] = []
                else:
                    params_r = []  # No fallback param inference
                rows = query(sql_r, tuple(params_r))
                LOGGER.info("[%s] executed(repaired) rows=%d", req_id, len(rows) if rows else 0)
                # overwrite the source SQL/params with repaired ones for downstream rendering
                sql = sql_r
                params = params_r
            except Exception as e2:
                return {"type": "db_error", "error": str(e2), "sql": repaired, "params": []}

        # 5) Sanitize + analyze (LLM-based, disclaimer-free)
        rows_sanitized = self._sanitize_rows(rows)
        result_text = self._analyze(user_query, sql, rows_sanitized)

        LOGGER.info("[%s] done row_count=%d", req_id, len(rows_sanitized))

        return {
            "type": "analysis",
            "source": sql,
            "params": params,
            "row_count": len(rows_sanitized),
            "rows": rows_sanitized,
            "result": result_text,
            "used_retrieval": False,
        }