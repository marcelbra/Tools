$nomen$ = "nomen2.lex"
$Morph$ = $nomen$ ({<N><Sg>}:{} | {<N><Pl>}:{en})
ALPHABET = [A-Za-z]
$Replace$ = ({een}:{en}) ^-> ()
$Morph$ || $Replace$