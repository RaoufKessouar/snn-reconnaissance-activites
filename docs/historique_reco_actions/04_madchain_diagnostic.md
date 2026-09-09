# Phase 2b — MAD-Chain : baseline et diagnostic d'instabilité

Primitives Walk(1)/Sit Down(3)/Fall(9), L=3 → 27 classes, T=40, hasard 0,037.

## Baseline
BNTT (100 participants) : **test 0,78** (~21x hasard, ordre perçu) MAIS validation
OSCILLE ±0,6 d'une epoch à l'autre, alors que le train converge à 0,97.
R-error 0,09 (ordre bien perçu). Erreurs : Sit/Fall pris pour Walk.

## Leviers testés (une variable à la fois)

| Levier | Attendu | Résultat | Verdict |
|---|---|---|---|
| SWA (moyennage poids) | lisser | val ada 0,85 · test std 0,665 / ada 0,731 | masque, AUCUN gain |
| Cosine LR | le modèle se pose | oscillation persiste | ÉCHEC → LR pas en cause |
| T=50 (+SWA) | + résolution | oscille PIRE (std/ada 0,08↔0,90), même plafond | écarté, verrouille T=40 |
| T=30 | densifier | trop grossier | écarté |
| Flip horizontal | régulariser | neutre / redondant | écarté |

## Conclusion du diagnostic
Aucune astuce d'entraînement ne supprime l'oscillation. Train lisse + val figée →
la SEULE source de variabilité restante est la NORMALISATION : chaque module BNTT
estime ses stats sur seulement B échantillons d'activations parcimonieuses → ne converge
jamais. Instabilité INTRINSÈQUE à BNTT → le remède doit viser la normalisation.

## Contrôle ANN vs SNN sur réel
ANN-BN : train PLAFONNE 0,40 · test std 0,088 / ada 0,151 (proche du hasard).
SNN : 0,78 (BNTT) → 0,975 (tdBN). Sur données réelles + split par sujet, l'ANN
S'EFFONDRE tandis que le SNN généralise → démonstration PLUS FORTE que sur le synthétique.
