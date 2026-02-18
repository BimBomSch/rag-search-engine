import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)
    return data["movies"]
