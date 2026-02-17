#!/usr/bin/env python3
"""Re-authenticate YouTube API with full scopes (upload + playlist management).

Run this directly in your terminal when the YouTube token expires or needs new scopes:
    python yt_auth.py

It will open a browser for Google OAuth consent, then save the token.
"""
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build as yt_build

CLIENT_SECRETS = Path(__file__).parent / "youtube_client_secrets.json"
TOKEN_FILE = Path(__file__).parent / "youtube_token.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

def main():
    if not CLIENT_SECRETS.exists():
        print("Error: youtube_client_secrets.json not found")
        print("Download OAuth2 Desktop App credentials from Google Cloud Console")
        return

    print("Opening browser for Google OAuth...")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
    creds = flow.run_local_server(port=8090, open_browser=True)

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print(f"Token saved to {TOKEN_FILE}")

    # Verify it works
    youtube = yt_build("youtube", "v3", credentials=creds)
    ch = youtube.channels().list(part="snippet", mine=True).execute()
    name = ch["items"][0]["snippet"]["title"] if ch.get("items") else "unknown"
    print(f"Authenticated as: {name}")

    # Test playlist access
    pl = youtube.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    print(f"Playlists: {len(pl.get('items', []))}")
    for p in pl.get("items", []):
        print(f"  - {p['snippet']['title']} ({p['id']})")

if __name__ == "__main__":
    main()
