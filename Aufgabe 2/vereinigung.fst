
% Quelle für alle folgenden Flexionsklassen: https://grammis.ids-mannheim.de/progr@mm/4064

% Definiert den Nominativ Plural durch Anhängen von "s" für Nomen mit Genus Femininum
% Neben dem Nominativ Plural ist für diese Flexionsklasse auch der Genitiv Plural abgedeckt
$nomen_s_pl$ = <N>:<> ({<Nom><Pl><F>}:{s} | {<Gen><Pl><F>}:{s} | {<Nom><Sg><F>}:{} | {<Gen><Sg><F>}:{})

% Definiert den Nominativ Plural durch Anhängen von "(e)n" für Nomen mit Genus Femininum
% Neben dem Nominativ Plural ist für diese Flexionsklasse auch der Genitiv Plural abgedeckt
$nomen_en_pl$ = <N>:<> ({<Nom><Sg><F>}:{} | {<Gen><Sg><F>}:{} | {<Nom><Pl><F>}:{en} | {<Gen><Pl><F>}:{en})


% Fängt Nomen ab, die auf "e" enden und daher nur noch durch "n" flektiert werden
% z.B. Katze/ Armee -> Katzeen/ Armeeen im ersten Transducer Schritt -> Katzen/ Armeen durch Anwenden der Replace Funktion
ALPHABET = [A-ZÄÖÜa-zäöü]
$Replace_een$ = ({een}:{en}) ^-> ()

$verben_flex$ = <V>:<> ({<pres><1><sg>}:{e} | {<pres><2><sg>}:{st} | {<pres><3><sg>}:{t} |\
{<pres><1><pl>}:{en} | {<pres><2><pl>}:{t} | {<pres><3><pl>}:{en})

% Ruft jeweils die Lexikondatei mit dem dazugehörigen Transducer auf
"nomen_s.lex" $nomen_s_pl$ |\
"nomen_en.lex" $nomen_en_pl$ || $Replace_een$ |\
"regulaere_verben.lex" $verben_flex$



