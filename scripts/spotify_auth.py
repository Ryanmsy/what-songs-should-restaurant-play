"""Client Credentials flow for the Spotify Web API — app-only token, no user login.

Used for pulling public catalog data (e.g. track popularity), as opposed to the
Authorization Code flow in deployment_draft/api/api_v2.py (/login, /callback),
which is for acting on behalf of a specific user.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
TOKEN_URL = "https://accounts.spotify.com/api/token"


def get_client_credentials_token() -> dict:
    """Exchange the app's client ID/secret for a short-lived access token."""
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set — add them to a .env file "
            "in the project root (see .env.example)"
        )

    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=10,
    )
    if not resp.ok:
        raise RuntimeError(f"Spotify token request failed: {resp.status_code} {resp.text}")

    data = resp.json()
    return {
        "access_token": data["access_token"],
        "expires_at": time.time() + data["expires_in"],
    }


if __name__ == "__main__":
    token = get_client_credentials_token()
    print(f"Got token, expires in {token['expires_at'] - time.time():.0f}s")
