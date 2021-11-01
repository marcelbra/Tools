% Liest Transducer ein
$Plural_s$ = "<plural_s.fst>"
$Plural_en$ = "<plural_en.fst>"
$Verb_inflect$ = "<regular_verb_inflection.fst>"

# Vereinigt die Transducer
$Unified_trans$ = $Plural_s$ | $Plural_en$ | $Verb_inflect$
$Unified_trans$