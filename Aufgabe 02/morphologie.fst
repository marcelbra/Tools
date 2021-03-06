% Abgabe Aufgabe 2: Marcel Braasch, Sinem Kühlewind, Nadja Seeberg

% Quelle für alle folgenden Flexionsklassen der Nomina: https://grammis.ids-mannheim.de/progr@mm/4064

% Definiert den Nominativ Plural durch Anhängen von "s" für Nomen mit Genus Femininum
% Neben dem Nominativ Plural ist für diese Flexionsklasse auch der Genitiv Plural abgedeckt
$nomen_s_pl$ = <N>:<> ({<Nom><Pl><F>}:{s} | {<Gen><Pl><F>}:{s} | {<Nom><Sg><F>}:{} | {<Gen><Sg><F>}:{})

% Definiert den Nominativ Plural durch Anhängen von "(e)n" für Nomen mit Genus Femininum
% Neben dem Nominativ Plural ist für diese Flexionsklasse auch der Genitiv Plural abgedeckt
$nomen_en_pl$ = <N>:<> ({<Nom><Sg><F>}:{} | {<Gen><Sg><F>}:{} | {<Nom><Pl><F>}:{en} | {<Gen><Pl><F>}:{en})


% Fängt Nomen ab, die auf "e" enden und daher nur noch durch "n" flektiert werden indem "een" durch "en" ersetzt wird
% z.B. Katze/ Armee -> Katzeen/ Armeeen im ersten Transducer Schritt -> Katzen/ Armeen durch Anwenden der Replace Funktion
ALPHABET = [A-ZÄÖÜa-zäöü]
$Replace_een$ = ({een}:{en}) ^-> ()

% Definiert die Präsenz-Konjugation regelmäßiger, bzw. schwacher Verben, welche regelmäßig durch Anhängen einer Flexionsendung
% konjugiert werden; Quelle: https://de.wiktionary.org/wiki/regelm%C3%A4%C3%9Figes_Verb
$verben_flex$ = <V>:<> ({<Pres><1><Sg>}:{e} | {<Pres><2><Sg>}:{st} | {<Pres><3><Sg>}:{t} |\
{<Pres><1><Pl>}:{en} | {<Pres><2><Pl>}:{t} | {<Pres><3><Pl>}:{en})

% Ruft jeweils die Lexikondatei mit dem dazugehörigen Transducer auf
"regulaere_verben.lex" $verben_flex$ |\
"nomen_s.lex" $nomen_s_pl$ |\
"nomen_en.lex" $nomen_en_pl$ || $Replace_een$




