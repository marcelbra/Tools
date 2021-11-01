% Liste einlesen
$nomen$ = "nomen2.lex"

% Definiert die Regel für Pluralinflektion
$Morph$ = $nomen$ ({<N><Sg>}:{} | {<N><Pl>}:{en})

% Stellt sicher, dass "doppel-e"s korrigiert werden.
ALPHABET = [A-Za-z]
$Replace$ = ({een}:{en}) ^-> ()

$Morph$ || $Replace$