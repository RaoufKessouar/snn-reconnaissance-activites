# Tableau global des résultats

## DVS-GC {0,7,8}, L=3, 27 classes
| Modèle | Train | Val |
|---|---|---|
| ANN-BN | 0,41 | 0,3464 |
| SNN (BNTT, Run A) | ~0,99 | 0,8196 |
Positions : 0,861 / 0,845 / 0,827. R-error pred 0,4538 / true 0,1846.

## MAD-Chain, L=3, 27 classes (test = 86-100)
| Config | Val | Test std | Test ada |
|---|---|---|---|
| BNTT baseline | 0,79 | 0,78 | 0,78 |
| BNTT + SWA | 0,85 | 0,665 | 0,731 |
| tdBN | 0,969 | 0,971 | 0,975 |
| tdBN fenêtre "late" | ~0,82 | — | — |
| ANN-BN | ~0,40 (train) | 0,088 | 0,151 |

## MAD-Chain, L=4, 81 classes
| Config | Val | Test |
|---|---|---|
| tdBN, 40 ep | 0,854 (sous-entraîné) | — |
| tdBN, 60 ep | 0,969 | ~0,97 |

## Détection de chute (v3, validation, ep43)
K=3 : recall 0,987 · précision 0,895 · F1 0,939 · fausses alarmes 0,124.

## Neurones (étude en cours)
QIF : bs=4 OOM (~44 Go), lancé bs=3. Résultats à compléter.
