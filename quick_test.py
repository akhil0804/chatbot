
# from db import query
# # print("Tables:", query("SHOW TABLES"))

# rows = query("SELECT name, item_code, item_name, stock_availability_status,stock_qty, base_price, supplier_list, description, qty, uom FROM `tabOpportunity Item` WHERE parent = 'AMS-ENQ-2022-00001' ")
# for r in rows:
#     print(r)


# test_azure_llm.py
from llm.openai_llm import AzureOpenAILLM
llm = AzureOpenAILLM()
print(llm.complete("Say 'hello'", system="You are a test.", temperature=0))