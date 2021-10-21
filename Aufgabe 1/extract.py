#!/usr/bin/env python

import re
import os
import sys
from html.parser import HTMLParser


class HTMLToText(HTMLParser):


    def __init__(self):
        super(HTMLToText, self).__init__()#)convert_charrefs=True)
        self.docs = []

    def dataloader(self,
                   path="./data/www.tagesschau.de/ausland/europa/italien-stichwahl-rom-103.html"
                   ):

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".html") and not file == "index.html":
                    with open(os.path.join(root, file)) as f:
                        html = f.read()
                        #print(" ".join(self.docs[0].plain_text.split()))
                        #self.docs.append(html)
                        return html

    def get_doc_by_path(self,
                        path="www.tagesschau.de/ausland/afrika/fossile-brennstoffe-unep-101.html"
                        ):
        with open(path) as f:
            html = f.read()
            return html

    def get_paragraph(self):
        html = self.get_doc_by_path()
        pattern = r"<p.*>(\n*.*?)<\/p>"
        paragraphs = re.findall(pattern, html)
        return paragraphs

    def get_headline(self):
        html = self.get_doc_by_path()
        title_pattern = r"\"headline\"\s:\s(.*),"
        title = re.findall(title_pattern, html)
        return title

    def get_description(self):
        html = self.get_doc_by_path()
        description_pattern = r"\"description\"\s:\s(.*),"
        description = re.findall(description_pattern, html)
        return description

    def save_text(self, file_name, title, description, paragraphs):
        with open(file_name, "w") as f:
            f.write(" ".join(title) + "\n"
                    + " ".join(description) + "\n"
                    + " ".join(paragraphs))



html_to_text = HTMLToText()
title = html_to_text.get_headline()
desc = html_to_text.get_description()
text = html_to_text.get_paragraph()
html_to_text.save_text("text.txt", title, desc, text)