# Opportunity Chatbot (Agents + Tools, No LLM/Gateway)

## 1) Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## 2) Configure DB
export MYSQL_HOST=127.0.0.1
export MYSQL_USER=root
export MYSQL_PASS=your_password
export MYSQL_DB=aquaerp
export MYSQL_PORT=3306

Update `config/db_schema.yaml` to match your actual columns.

## 3) Run
uvicorn app:app --reload

## 4) Test
POST http://localhost:8000/chat
Body:
{
  "message": "What are the items available in opportunity OPP-00123?"
}

or

{
  "message": "For opportunity OPP-00123, how many items are there and give me quoted suppliers and contact details"
}

or provide the ID via context:
{
  "message": "Give me quoted suppliers and contact details",
  "context": { "opportunity_id": "OPP-00123" }
}