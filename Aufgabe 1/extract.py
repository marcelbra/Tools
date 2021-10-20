#!/usr/bin/env python

import re
import os
import sys
from html.parser import HTMLParser


class HTMLToText(HTMLParser):
    plain_text = ""

    def handle_data(self, data):
        # Filtern nach <title> und <p>? Index Files ausschließen?
        if self.correct_tag:
            self.plain_text += data


html_to_text = HTMLToText()

file_path = "./data/aufgabe1/www.tagesschau.de/"

for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if file.endswith("fluechtlinge-101.html") and not file == "index.html":
            with open(os.path.join(root, file)) as f:
                html = f.read()
                html_to_text.feed(html)

print(" ".join(html_to_text.plain_text.split()))