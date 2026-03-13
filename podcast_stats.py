#!/usr/bin/env python3
"""
Podcast Stats — Aggregate reviews, comments, likes, and analytics from all platforms.

Usage:
    python podcast_stats.py              # All platforms
    python podcast_stats.py --youtube    # YouTube only
    python podcast_stats.py --apple      # Apple Podcasts only
    python podcast_stats.py --spotify    # Spotify only
    python podcast_stats.py --castopod   # Castopod downloads only
    python podcast_stats.py --comments   # Include full YouTube comments
    python podcast_stats.py --json       # Output as JSON
    python podcast_stats.py --json --upload  # Output JSON and upload to BunnyCDN
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

YOUTUBE_PLAYLIST = "PLGq4uZyNV1yYH_rcitTTPVysPbC6-7pe-"
APPLE_PODCAST_ID = "1875205848"
APPLE_STOREFRONTS = ["us", "gb", "ca", "au"]
SPOTIFY_SHOW_ID = "0ZrpMigG1fo0CCN7F4YmuF"
NAS_SSH = "luke@mmgnas-10g"
NAS_SSH_PORT = "8001"
DOCKER_BIN = "/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
CASTOPOD_DB_CONTAINER = "castopod-mariadb-1"

BUNNY_STORAGE_ZONE = "lukeattheroost"
BUNNY_STORAGE_KEY = os.getenv("BUNNY_STORAGE_KEY", "")
BUNNY_STORAGE_REGION = "la"
BUNNY_ACCOUNT_KEY = os.getenv("BUNNY_ACCOUNT_KEY", "")


def _find_ytdlp():
    """Find yt-dlp: check local venv first, then fall back to PATH."""
    import shutil
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "yt-dlp")
    if os.path.exists(venv_path):
        return venv_path
    path_bin = shutil.which("yt-dlp")
    if path_bin:
        return path_bin
    return "yt-dlp"


def gather_apple_reviews():
    all_reviews = []
    seen_ids = set()

    for storefront in APPLE_STOREFRONTS:
        url = f"https://itunes.apple.com/{storefront}/rss/customerreviews/id={APPLE_PODCAST_ID}/sortby=mostrecent/json"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        feed = data.get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            continue

        for entry in entries:
            if "im:name" in entry and "im:rating" not in entry:
                continue

            review_id = entry.get("id", {}).get("label", "")
            if review_id in seen_ids:
                continue
            seen_ids.add(review_id)

            author = entry.get("author", {}).get("name", {}).get("label", "Unknown")
            title = entry.get("title", {}).get("label", "")
            content = entry.get("content", {}).get("label", "")
            rating = int(entry.get("im:rating", {}).get("label", "0"))
            updated = entry.get("updated", {}).get("label", "")
            date_str = updated[:10] if updated else ""

            all_reviews.append({
                "author": author,
                "title": title,
                "content": content,
                "rating": rating,
                "date": date_str,
                "storefront": storefront.upper(),
            })

    avg_rating = round(sum(r["rating"] for r in all_reviews) / len(all_reviews), 1) if all_reviews else None
    return {
        "avg_rating": avg_rating,
        "review_count": len(all_reviews),
        "reviews": all_reviews[:10],
    }


def gather_spotify():
    result = {"show_title": None, "rating": None, "url": f"https://open.spotify.com/show/{SPOTIFY_SHOW_ID}"}

    try:
        oembed_url = f"https://open.spotify.com/oembed?url=https://open.spotify.com/show/{SPOTIFY_SHOW_ID}"
        resp = requests.get(oembed_url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result["show_title"] = data.get("title")

        show_url = f"https://open.spotify.com/show/{SPOTIFY_SHOW_ID}"
        resp = requests.get(show_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })

        rating_match = re.search(r'"ratingValue"\s*:\s*"?([\d.]+)"?', resp.text)
        if rating_match:
            result["rating"] = float(rating_match.group(1))
        else:
            rating_match2 = re.search(r'rating["\s:]*(\d+\.?\d*)\s*/\s*5', resp.text, re.IGNORECASE)
            if rating_match2:
                result["rating"] = float(rating_match2.group(1))
    except Exception:
        pass

    return result


def gather_youtube(include_comments=False):
    result = {
        "total_views": 0,
        "total_likes": 0,
        "total_comments": 0,
        "subscribers": None,
        "videos": [],
    }

    try:
        proc = subprocess.run(
            [_find_ytdlp(), "--dump-json", "--flat-playlist",
             f"https://www.youtube.com/playlist?list={YOUTUBE_PLAYLIST}"],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return result

    video_ids = []
    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            vid = entry.get("id") or entry.get("url", "").split("=")[-1]
            if vid:
                video_ids.append(vid)
        except json.JSONDecodeError:
            continue

    if not video_ids:
        return result

    total_views = 0
    total_likes = 0
    total_comments = 0
    videos = []

    for vid in video_ids:
        try:
            cmd = [_find_ytdlp(), "--dump-json", "--no-download", f"https://www.youtube.com/watch?v={vid}"]
            if include_comments:
                cmd.insert(2, "--write-comments")
            vr = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if vr.returncode != 0:
                continue
            vdata = json.loads(vr.stdout)

            title = vdata.get("title", "Unknown")
            views = vdata.get("view_count", 0) or 0
            likes = vdata.get("like_count", 0) or 0
            comment_count = vdata.get("comment_count", 0) or 0
            upload_date = vdata.get("upload_date", "")
            if upload_date:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"

            comments_list = []
            if include_comments:
                for c in (vdata.get("comments") or [])[:5]:
                    comments_list.append({
                        "author": c.get("author", "Unknown"),
                        "text": c.get("text", "")[:200],
                        "time": c.get("time_text", ""),
                        "likes": c.get("like_count", 0),
                    })

            total_views += views
            total_likes += likes
            total_comments += comment_count

            videos.append({
                "title": title,
                "views": views,
                "likes": likes,
                "comments": comment_count,
                "date": upload_date,
            })
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            continue

    # Get subscriber count
    if videos:
        try:
            vr = subprocess.run(
                [_find_ytdlp(), "--dump-json", "--no-download", "--playlist-items", "1",
                 f"https://www.youtube.com/playlist?list={YOUTUBE_PLAYLIST}"],
                capture_output=True, text=True, timeout=30
            )
            if vr.returncode == 0:
                ch_data = json.loads(vr.stdout)
                sub = ch_data.get("channel_follower_count")
                if sub is not None:
                    result["subscribers"] = sub
        except Exception:
            pass

    result["total_views"] = total_views
    result["total_likes"] = total_likes
    result["total_comments"] = total_comments
    result["videos"] = videos
    return result


def _run_db_query(sql):
    # If running on NAS (docker socket available), exec directly
    docker_bin = None
    for path in [DOCKER_BIN, "/usr/bin/docker", "/usr/local/bin/docker"]:
        if os.path.exists(path):
            docker_bin = path
            break

    db_pass = os.getenv("CASTOPOD_DB_PASS", "")
    if docker_bin:
        # Pass password via MYSQL_PWD env var instead of command line (not visible in ps)
        cmd = [docker_bin, "exec", "-i", "-e", f"MYSQL_PWD={db_pass}",
               CASTOPOD_DB_CONTAINER,
               "mysql", "-u", "castopod", "castopod", "-N"]
    else:
        cmd = [
            "ssh", "-p", NAS_SSH_PORT, NAS_SSH,
            f"{DOCKER_BIN} exec -i -e MYSQL_PWD={db_pass} {CASTOPOD_DB_CONTAINER} mysql -u castopod castopod -N"
        ]
    try:
        proc = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=30)
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        if proc.returncode != 0 and not stdout:
            return None, stderr
        return stdout, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def gather_castopod():
    result = {"total_downloads": 0, "unique_listeners": 0, "episodes": []}

    query = (
        "SELECT p.title, "
        "(SELECT SUM(hits) FROM cp_analytics_podcasts WHERE podcast_id = p.id), "
        "(SELECT SUM(unique_listeners) FROM cp_analytics_podcasts WHERE podcast_id = p.id) "
        "FROM cp_podcasts p WHERE p.handle = 'LukeAtTheRoost' LIMIT 1;"
    )
    episode_query = (
        "SELECT e.title, e.slug, COALESCE(SUM(ae.hits), 0), e.published_at "
        "FROM cp_episodes e LEFT JOIN cp_analytics_podcasts_by_episode ae ON ae.episode_id = e.id "
        "WHERE e.podcast_id = (SELECT id FROM cp_podcasts WHERE handle = 'LukeAtTheRoost') "
        "GROUP BY e.id ORDER BY e.published_at DESC;"
    )

    out, err = _run_db_query(query)
    if err or not out:
        return result

    parts = out.split("\t")
    if len(parts) >= 3:
        result["total_downloads"] = int(parts[1]) if parts[1] and parts[1] != "NULL" else 0
        result["unique_listeners"] = int(parts[2]) if parts[2] and parts[2] != "NULL" else 0
    elif len(parts) >= 2:
        result["total_downloads"] = int(parts[1]) if parts[1] and parts[1] != "NULL" else 0

    out, err = _run_db_query(episode_query)
    if err or not out:
        return result

    for line in out.strip().split("\n"):
        cols = line.split("\t")
        if len(cols) >= 4:
            result["episodes"].append({
                "title": cols[0],
                "downloads": int(cols[2]) if cols[2] else 0,
                "date": cols[3][:10] if cols[3] else "",
            })

    return result


def print_apple(data):
    print("\n⭐ APPLE PODCASTS")
    print("─" * 40)
    if data["reviews"]:
        print(f"  Rating: {data['avg_rating']}/5 ({data['review_count']} reviews)")
        print()
        for r in data["reviews"]:
            stars = "★" * r["rating"] + "☆" * (5 - r["rating"])
            print(f"  {stars} \"{r['title']}\" — {r['author']} ({r['date']}, {r['storefront']})")
            if r["content"] and r["content"] != r["title"]:
                content_preview = r["content"][:120]
                if len(r["content"]) > 120:
                    content_preview += "..."
                print(f"    {content_preview}")
    else:
        print("  No reviews found")


def print_spotify(data):
    print("\n🎵 SPOTIFY")
    print("─" * 40)
    if data["show_title"]:
        print(f"  Show: {data['show_title']}")
    if data["rating"]:
        print(f"  Rating: {data['rating']}/5")
    else:
        print("  Rating: Not publicly available (Spotify hides ratings from web)")
    print(f"  Link: {data['url']}")


def print_youtube(data):
    print("\n📺 YOUTUBE")
    print("─" * 40)
    sub_str = f" | Subscribers: {data['subscribers']:,}" if data["subscribers"] else ""
    print(f"  Total views: {data['total_views']:,} | Likes: {data['total_likes']:,} | Comments: {data['total_comments']:,}{sub_str}")
    print()
    for v in data["videos"]:
        print(f"  {v['title']}")
        print(f"    {v['views']:,} views, {v['likes']:,} likes, {v['comments']:,} comments — {v['date']}")


def print_castopod(data):
    print("\n📊 DOWNLOADS (Castopod)")
    print("─" * 40)
    print(f"  Total downloads: {data['total_downloads']:,} | Unique listeners: {data['unique_listeners']:,}")
    if data["episodes"]:
        print()
        for ep in data["episodes"]:
            print(f"  {ep['title']} — {ep['downloads']:,} downloads ({ep['date']})")


def upload_to_bunnycdn(json_data):
    storage_url = f"https://{BUNNY_STORAGE_REGION}.storage.bunnycdn.com/{BUNNY_STORAGE_ZONE}/stats.json"
    resp = requests.put(
        storage_url,
        data=json_data,
        headers={
            "AccessKey": BUNNY_STORAGE_KEY,
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()

    purge_url = "https://api.bunny.net/purge"
    requests.post(
        purge_url,
        params={"url": "https://cdn.lukeattheroost.com/stats.json"},
        headers={"AccessKey": BUNNY_ACCOUNT_KEY},
        timeout=15,
    )
    print("Uploaded stats.json to BunnyCDN and purged cache", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Podcast analytics aggregator")
    parser.add_argument("--youtube", action="store_true", help="YouTube only")
    parser.add_argument("--apple", action="store_true", help="Apple Podcasts only")
    parser.add_argument("--spotify", action="store_true", help="Spotify only")
    parser.add_argument("--castopod", action="store_true", help="Castopod only")
    parser.add_argument("--comments", action="store_true", help="Include YouTube comments")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    parser.add_argument("--upload", action="store_true", help="Upload JSON to BunnyCDN (requires --json)")
    args = parser.parse_args()

    if args.upload and not args.json_output:
        print("Error: --upload requires --json", file=sys.stderr)
        sys.exit(1)

    run_all = not (args.youtube or args.apple or args.spotify or args.castopod)

    results = {}
    if run_all or args.castopod:
        results["castopod"] = gather_castopod()
    if run_all or args.apple:
        results["apple"] = gather_apple_reviews()
    if run_all or args.spotify:
        results["spotify"] = gather_spotify()
    if run_all or args.youtube:
        results["youtube"] = gather_youtube(include_comments=args.comments or args.youtube)

    if args.json_output:
        output = {
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **results,
        }
        json_str = json.dumps(output, indent=2, ensure_ascii=False)
        print(json_str)
        if args.upload:
            upload_to_bunnycdn(json_str)
    else:
        print("=" * 45)
        print("  PODCAST STATS: Luke at the Roost")
        print("=" * 45)
        if "castopod" in results:
            print_castopod(results["castopod"])
        if "apple" in results:
            print_apple(results["apple"])
        if "spotify" in results:
            print_spotify(results["spotify"])
        if "youtube" in results:
            print_youtube(results["youtube"])
        print()


if __name__ == "__main__":
    main()
