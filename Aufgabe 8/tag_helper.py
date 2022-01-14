replaceable = [".", ",", ":", "``", "-LRB-", "-RRB-"]
replace_with = ["DOT", "COM", "COL", "ACC", "-LRB-TAG", "-BRB-TAG"]

def restore_replaced_tag(constituent):
    tag = constituent[0]
    if tag in replace_with:
        constituent = (map_tag_to_word(tag), *constituent[1:])
    s = 0
    return constituent

def map_tag_to_word(tag):
    """Inverse mapping of `map_word_to_tag`."""
    for _replace, _with in zip(replaceable, replace_with):
        tag = tag.replace(_with, _replace)
    return tag

def map_word_to_tag(word):
    """Maps a word which has the same appearance as its tag to
     an artificially constructed tag, e.g. '.' -> 'DOT'. We
     exchange this to better distinguish between words and tags."""
    for _replace, _with in zip(replaceable, replace_with):
        word = word.replace(_replace, _with)
    return word

def substitute_tags(tokens):
    for i in range(len(tokens)):
        current_token = tokens[i]
        next_token = tokens[i+1]
        if (current_token==next_token
        and current_token not in [")", "("]):
            for _replace, _with in zip(replaceable, replace_with):
                tokens[i] = tokens[i].replace(_replace, _with)
        if i==len(tokens)-2:
            break
    return tokens

#