# debug_parser.py
from unstructured.partition.auto import partition

pdf_path = "/users/archerkim/CLionProjects/MyAI/dropbox/The_Feynman_Lectures.pdf"

print("starting partition...")
try:
    elements = partition(filename=pdf_path)
    print(f"successfully parsed {len(elements)} elements.")
except Exception as e:
    print(f"an error occurred: {e}")