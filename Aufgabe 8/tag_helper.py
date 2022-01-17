d = {'PRN': 'PRN', 'JJR': 'JJR', 'INTJ': 'INTJ', 'NNS': 'NNS',
     'CC': 'CC', 'X': 'X', 'DT': 'DT', 'UCP': 'UCP', 'PDT': 'PDT',
     'WDT': 'WDT', 'MD': 'MD', 'RP': 'RP', 'WHADVP': 'WHADVP',
     'ADJP': 'ADJP', 'LST': 'LST', 'RBS': 'RBS', '``': '``',
     'NX': 'NX', 'SQ': 'SQ', 'WP$': 'WP$', 'FW': 'FW', 'SBAR': 'SBAR',
     'WHPP': 'WHPP', 'POS': 'POS', '.': '.-TAG', 'VP': 'VP',
     'WRB': 'WRB', 'WP': 'WP', 'NP': 'NP', 'PRT': 'PRT', 'EX': 'EX',
     'VBZ': 'VBZ', 'CD': 'CD', 'WHNP': 'WHNP', 'RBR': 'RBR', 'QP': 'QP',
     '-LRB-': '-LRB--TAG', 'CONJP': 'CONJP', '-RRB-': '-RRB-', 'NN': 'NN',
     'S': 'S', 'FRAG': 'FRAG', 'TOP': 'TOP', 'VBD': 'VBD', 'PRP$': 'PRP$',
     ':': ':', 'SINV': 'SINV', 'RRC': 'RRC', 'JJ': 'JJ', 'VBG': 'VBG',
     'VBP': 'VBP', 'PP': 'PP', 'UH': 'UH', 'NNPS': 'NNPS', 'LS': 'LS',
     'NAC': 'NAC', 'VB': 'VB', 'TO': 'TO', 'WHADJP': 'WHADJP', 'ADVP': 'ADVP',
     'SBARQ': 'SBARQ', 'RB': 'RB', '#': '#', 'PRP': 'PRP', 'IN': 'IN', "''": "''",
     'JJS': 'JJS', 'NNP': 'NNP', 'SYM': 'SYM', ',': ',-TAG',
     'VBN': 'VBN', '$': '$-TAG'}

def map_tag_to_modified_tag(tokens):
    for i in range(len(tokens)):
        current_token = tokens[i]
        next_token = tokens[i+1]
        """        
        Offensichtlich ist das mappen relativ aufwändig und ist feature engineered.
        Das Problem war, dass wir zu spät damit angefangen haben andere samples zu testen.
        Wir dachten es genügt "." auf ".-TAG", "$" auf "$-TAG" etc. zu mappen. Später
        ist uns dann aufgefallen, dass es noch weitere Ausnahmen gibt. Wir haben alle
        abgedeckt, aber wie gesagt, es wäre sicherlich einfacher gegangen.        
        """
        if (
            (current_token==next_token and current_token not in [")", "("])
            or
            (current_token=="." and next_token in ["?", "!"])
            or
            (current_token == "-LRB-" and next_token in ["-LCB-"])
            or
            (current_token == "$" and next_token in ["US$", "C$", "HK$", "HK$", "NZ$", "A$", "C", "M$", "S$"])
            ):
            tokens[i] = tag_to_modified_tag(tokens[i])
        if i==len(tokens)-2:
            break
    return tokens

def tag_to_modified_tag(tag):
    if tag in d:
        return d[tag]
    else:
        return tag

def modified_tag_to_tag(modified):
    rev_d = {value:key for key,value in d.items()}
    if modified in rev_d:
        return rev_d[modified]
    else:
        return modified