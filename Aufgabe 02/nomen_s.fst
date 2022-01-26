% Liste einlesen
$nomen$ = "nomen.lex"

% Definiert die Regel für Pluralinflektion
$MORPH$ = $nomen$ ({<N><Pl><F>}:{s} | {<N><Sg>}:{})
$MORPH$

