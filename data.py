import re
import os
from urllib.request import urlopen
import bz2

import requests
import mwxml
import mwparserfromhell

def get_wikidump(lang):

    compressed_filename = f"{lang}wiki-latest-pages-articles.xml.bz2"
    if not os.path.exists(compressed_filename):
        print("Downloading file...")
        url = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
        response = requests.get(url)
        with open(compressed_filename, 'wb') as output_file:
            output_file.write(response.content)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

    decompressed_filename = f"{lang}wiki-latest-pages-articles.xml"
    if not os.path.exists(decompressed_filename):
        print("Decompressing file...")
        with bz2.BZ2File(compressed_filename, 'rb') as f_in, open(decompressed_filename, 'wb') as f_out:
            for data in iter(lambda : f_in.read(100 * 1024), b''):
                f_out.write(data)
        print("File decompressed successfully.")
    else:
        print("Decompressed file already exists.")


def main():
    lang = "simple"
    get_wikidump(lang)
    with open(f"{lang}wiki-latest-pages-articles.xml") as file:
        dump = mwxml.Dump.from_file(file)
        for page in dump:
            print(f"Parsing {page.title}...")
            for revision in page:
                text = revision.text
                if text is not None:
                    wikicode = mwparserfromhell.parse(text)
                    for template in wikicode.filter_templates():
                        try:
                            wikicode.remove(template)
                        except ValueError:
                            pass
                    for link in wikicode.filter_wikilinks():
                        if link.title.strip_code().startswith('File:'):
                            try:
                                wikicode.remove(link)
                            except ValueError:
                                pass
                    for heading in wikicode.filter_headings():
                        wikicode.remove(heading)
                    plaintext = wikicode.strip_code()
                    plaintext = re.sub("[ \t]*\n+[ \t]*", "\n", plaintext).strip()
                    plaintext = "\n".join([line for line in plaintext.split("\n") if len(line) >= 80])
                    if plaintext:
                        with open(f"{lang}wiki.txt", "a+") as outfile:
                            outfile.write(f"{plaintext}\n")

if __name__ == "__main__":
    main()
