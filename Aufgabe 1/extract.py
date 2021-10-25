"""
P7 Tools - Aufgabe 1
HTML Extractor
"""

import re
import os
import sys

class HTMLToText:

    def __init__(self):
        """
        Expects a corpus of Tagesschau html documents.
        Iterates through each document and extracts
        H1 heading, h2 headings, the description and paragraphs.
        Makes sure the order is preserved (especially h2s and paragraphs).

        How to run:
        $ python3 extract.py path_to_dir > text.txt
        """
        self.docs = []
        self.doc = None
        self.result = ""

    def load_data(self):
        """
        Runs trough the directory specified by arg[1].
        Saves the respective html files.
        In PyCharm: Run -> Edit Configuations -> Parameters ->
        """
        for root, dirs, files in os.walk(sys.argv[1]):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    self.docs.append(f.read())

    def iterate(self):
        """
        Iterates through each html file stored.
        Extracts text by calling respective method for each of the entities.
        Passes the information on to format the contents.
        Does some cleaning post-processing.
        When done, moves on to write to text file.
        """
        assert len(self.docs) > 0, "You need to call load_data first!"
        for doc in self.docs:
            self.doc = doc
            try:
                paragraphs = self.get_paragraphs()
                indices = self.get_indices_of(paragraphs)
                title, description, h2s = self.get_contents()
            except IndexError:
                # This happes when a document is processed which is not a typical
                # article, for example one which references a collection of articles.
                # We simply skip these.
                continue
            self.format_doc(title,
                           description,
                           paragraphs,
                           h2s,
                           indices)
        self.clean_text()
        self.write_doc()

    def clean_text(self):
        """"
        Discards special characters that were created in preprocessing.
        Deletes URLs embedded in text.
        Deletes any spare/ left over tags.
        """
        self.result = self.result.replace("\\", '')
        url_begin_pattern = r"<\w.*>\S"  # Position is +1 of desired index
        closing_tag_pattern = r"</\w.*>"
        url_begin = re.finditer(url_begin_pattern, self.result)
        closing_tag = re.finditer(closing_tag_pattern, self.result)
        if url_begin is not None:
            for idx_begin in url_begin:
                self.result = self.result.replace(idx_begin.group()[:-1], "")
        if closing_tag is not None:
            for tag in closing_tag:
                self.result = self.result.replace(tag.group(), "")
        return self.result

    def write_doc(self):
        """
        Writes the document to text file.
        """
        with open("text.txt", "w") as f:
            f.write(self.result)


    def format_doc(self, heading, description, paragraphs, h2s, indices):
        """
        Determines the format of the output text.
        Formatted variables (each one containing text) are saved.
        When scraping, in indices we saved which paragraph (by index) had an h2
        heading in front of it. This way, when creating the running text, we
        know exactly when to insert the next h2 heading.
        """
        self.result += f"{heading[1:-1]}\n\n{description[1:-1]}\n"  # Remove " .. " in beginning and end
        for i, paragraph in enumerate(paragraphs):
            # Check if there is an h2 heading before the current paragraph
            if i in indices:
                self.result += f"{h2s[indices.index(i)]}\n"
            self.result += f"{paragraph}\n"
        self.result += "\n" * 2

    def get_paragraphs(self):
        """
        Retrieves the paragraphs of the current docs.
        """
        # Find all paragraphs and post process
        pattern = r"<p class=\"m.*>(\n*.*?)<\/p>"
        all_paragraphs = re.findall(pattern, self.doc)
        all_paragraphs = [x.replace("\n", "") for x in all_paragraphs][1:-1]
        return all_paragraphs

    def get_indices_of(self, paragraphs):
        """
        Determin if a paragraph has an h2 heading in front of it.
        """
        # Determines which of the paragraphs have an h2 heading above them
        position_pattern = r"<h2(?:.|\n)*?>.*(?:.|\n)*?<.*>\n(.*)</p>"
        h2_paragraphs = re.findall(position_pattern, self.doc)
        indices = [paragraphs.index(p) if p in paragraphs else None for p in h2_paragraphs]
        return indices


    def get_contents(self):
        """
        Retrieves the contents of the current doc.
        The patterns are: 1) title 2) description 3) h2 heading
        :return: 3-tuple
            title: str
            description: str
            h2s: List[str]
        """
        patterns = [r"\"headline\"\s:\s(.*),",  # title
                    r"\"description\"\s:\s(.*),",  # description
                    r"<h2.*?>(.*)</h2>"]  # h2s
        contents = [self.get_content_with(pattern) for pattern in patterns]

        # Extract from regex-list: title and desc. are unique, h2s are List[str], thus we grab these from the lists
        contents = [contents[i][0] if i < 2 else contents[i]
                    for i, _ in enumerate(contents)]
        return contents

    def get_content_with(self, pattern):
        """
        Returns all matches of the current doc.
        :param pattern: The pattern to match against .
        :return: List of matches.
        """
        return re.findall(pattern, self.doc)


html_to_text = HTMLToText()
html_to_text.load_data()
html_to_text.iterate()
