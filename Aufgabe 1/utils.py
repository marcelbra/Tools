"""
Replace dots in abreviations with QUAK.
"""

with open("abbreviations.txt", "r", encoding="utf-8") as f:
    text = f.read()

with open("abbreviations_mod.txt", "w", encoding="utf-8") as f:
    f.write(text.replace(".", "QU4K"))