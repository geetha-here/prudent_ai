# movies_client.py

import requests
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

API_KEY = os.getenv("MOVIE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing environment variable: MOVIE_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
TIMEOUT = 4
RETRIES = 1


def get_director(movie_id: int) -> str | None:
    """
    Fetch the director from TMDB credits API.
    """
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": API_KEY}

    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            for crew_member in data.get("crew", []):
                if crew_member.get("job") == "Director":
                    return crew_member.get("name")

            return None
        except Exception:
            if attempt == RETRIES:
                raise


def search_movies(query: str, page: int):
    """
    Search movie titles + fetch director for each movie.
    """
    # If q is empty → return empty results
    if not query:
        return {
            "results": [],
            "total_results": 0,
            "total_pages": 0,
            "page": page,
        }

    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": query,
        "page": page,
    }

    for attempt in range(RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            movies = data.get("results", [])

            results = []
            for movie in movies:
                movie_id = movie.get("id")
                director = get_director(movie_id) if movie_id else None

                results.append({
                    "title": movie.get("title"),
                    "director": director,
                })

            return {
                "results": results,
                "total_results": data.get("total_results", 0),
                "total_pages": data.get("total_pages", 0),
                "page": page,
            }

        except Exception:
            if attempt == RETRIES:
                raise
