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
                   path="/Users/marcelbraasch/Desktop/Tools"
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
                        path="/Users/marcelbraasch/Desktop/Tools/www.tagesschau.de/wirtschaft/verbraucher/tanktourismus-eu-benzinpreise-101.html"
                        ):
        with open(path) as f:
            html = f.read()
            return html

    def get_paragraph(self):
        html = self.get_doc_by_path()
        pattern = r"(.*)</p>"
        paragraphs = re.findall(pattern, html)
        filtered = []
        for paragraph in paragraphs:


        s = 0



html_to_text = HTMLToText()
html_to_text.get_paragraph()
