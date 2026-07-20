"""
Loading .env here (not in each script's __main__) means every entry
point -- fetch.py, embed.py, agent.py, and eventually the FastAPI app --
gets environment variables from .env automatically just by importing
anything from the pipeline package, with no risk of forgetting to wire
it into a new script later.
"""

from dotenv import load_dotenv

load_dotenv()