import os

# Set before app.py runs load_dotenv(): a local .env with a real key must never be used in tests.
os.environ["TRIAGE_USE_LLM"] = "false"
os.environ["SEED_SAMPLE_TICKETS"] = "false"
