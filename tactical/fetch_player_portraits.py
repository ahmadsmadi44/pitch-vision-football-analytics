"""Fetch reusable player thumbnails and retain source attribution metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests


PLAYERS = {
    "LIV-1": "Alisson Becker", "LIV-66": "Trent Alexander-Arnold", "LIV-5": "Ibrahima Konate",
    "LIV-4": "Virgil van Dijk", "LIV-26": "Andrew Robertson", "LIV-3": "Fabinho",
    "LIV-14": "Jordan Henderson", "LIV-6": "Thiago Alcantara", "LIV-11": "Mohamed Salah",
    "LIV-10": "Sadio Mane", "LIV-23": "Luis Diaz",
    "RMA-1": "Thibaut Courtois", "RMA-2": "Dani Carvajal", "RMA-3": "Eder Militao",
    "RMA-4": "David Alaba", "RMA-23": "Ferland Mendy", "RMA-14": "Casemiro",
    "RMA-10": "Luka Modric", "RMA-8": "Toni Kroos", "RMA-15": "Federico Valverde",
    "RMA-9": "Karim Benzema", "RMA-20": "Vinicius Junior",
}


def main():
    destination = Path("pitch-vision-platform/frontend/public/assets/players")
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PitchVisionPortfolio/1.0 (local educational project)"
    credits_path = destination / "portrait-credits.json"
    try:
        credits = json.loads(credits_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        credits = {}
    for player_id, title in PLAYERS.items():
        path = destination / f"{player_id}.jpg"
        if path.exists():
            credits.setdefault(player_id, {
                "player": title,
                "page": "https://www.thesportsdb.com/",
                "sourceImage": None,
                "notice": "Player media supplied by TheSportsDB. Review provider terms before public redistribution.",
            })
            print(f"cached portrait: {player_id} {title}")
            continue
        response = session.get("https://www.thesportsdb.com/api/v1/json/3/searchplayers.php", params={"p": title}, timeout=30)
        response.raise_for_status()
        matches = response.json().get("player") or []
        page = next((item for item in matches if item.get("strSport") == "Soccer"), None)
        thumbnail = (page or {}).get("strCutout") or (page or {}).get("strThumb")
        if not thumbnail:
            print(f"missing portrait: {player_id} {title}")
            continue
        image = session.get(thumbnail, timeout=30)
        image.raise_for_status()
        path.write_bytes(image.content)
        credits[player_id] = {
            "player": title,
            "page": f'https://www.thesportsdb.com/player/{page.get("idPlayer")}',
            "sourceImage": thumbnail,
            "notice": "Player media supplied by TheSportsDB. Review provider terms before public redistribution.",
        }
        credits_path.write_text(json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{player_id}: {len(image.content)} bytes")
        time.sleep(.2)
    credits_path.write_text(json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"downloaded {len(credits)}/{len(PLAYERS)} portraits")


if __name__ == "__main__":
    main()
