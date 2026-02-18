from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


MOVIES = "./data/movies.json"

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    results = []
    for movie in movies:
        if query in movie["title"]:
            results.append(movie)
            if len(results) >= limit:
                break
    return results