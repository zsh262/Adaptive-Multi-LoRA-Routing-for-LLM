import requests
from typing import Optional

BASE_URL = "http://127.0.0.1:8000"


class MultiLoraClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def ask(
        self,
        query: str,
        domain: Optional[str] = "auto",
        route_available_only: bool = False,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/api/llm/ask",
            json={
                "query": query,
                "domain": domain,
                "route_available_only": route_available_only,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def ask_coding(self, query: str) -> dict:
        return self.ask(query, domain="coding")

    def ask_paper(self, query: str) -> dict:
        return self.ask(query, domain="paper")

    def ask_speech(self, query: str) -> dict:
        return self.ask(query, domain="speech")

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False