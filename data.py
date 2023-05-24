import os
import re
import time

import requests

visited_pages_file = "visited_pages.txt"
visited_pages = set()
if os.path.exists(visited_pages_file):
    with open(visited_pages_file, "r") as f:
        visited_pages = set(int(line.strip()) for line in f)
while True:
    start_time_outer = time.time()
    try:
        random_articles = requests.get("https://simple.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=500").json()
        for article in random_articles["query"]["random"]:
            if article["id"] in visited_pages:
                continue
            visited_pages.add(article["id"])
            start_time_inner = time.time()
            try:
                page_content = requests.get(f"https://simple.wikipedia.org/w/api.php?action=query&format=json&titles={article['title']}&prop=extracts&explaintext").json()
                page_text = list(page_content["query"]["pages"].values())[0]["extract"]
                page_text = re.sub(r"==[^=]+?==", "\n", page_text)
                page_text = page_text.replace("\\n", "\n")
                page_text = re.sub(r"\n+", "\n", page_text).strip()
                with open("input.txt", "a+", encoding="utf-8") as f:
                    for line in page_text.split("\n"):
                        if len(line) >= 80:
                            f.write(f"{line}\n")
                with open(visited_pages_file, "a+") as f:
                    f.write(str(article["id"]) + "\n")
            except Exception as e:
                print(f"Error occurred with page {article['title']}: {e}")
            sleep_time_inner = max(3 - (time.time() - start_time_inner), 0)
            time.sleep(sleep_time_inner)
    except Exception as e:
        print(f"Error occurred fetching random articles: {e}")
    sleep_time_outer = max(3 - (time.time() - start_time_outer), 0)
    time.sleep(sleep_time_outer)
