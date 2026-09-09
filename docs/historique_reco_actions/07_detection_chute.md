# Phase 3 — Détection de chute (application, EN COURS)

## Reformulation
De « classer une chaîne » à « DÉTECTER + LOCALISER » la chute dans un flux. Vocabulaire
= 9 activités ; séquences de 5 ; chute présente/absente (~50 %), position aléatoire, au
plus une ; lecture PAR IMAGE ([B,T,9]) ; loss pondérée (chute rehaussée). Base tdBN,
split par sujet.

## Métriques et décision
Rappel (sensibilité), taux de fausses alarmes, F1, localisation. Décision : chute déclarée
si >= K images-chute (K = point de fonctionnement ; compromis rappel/fausses alarmes).

## Runs

| Version | Config | Résultat | Note |
|---|---|---|---|
| v1 | T40, fb4, 40 ep | recall ~1,0 mais FA ~1,0 ; F1<=0,81 | trop agressif (boost) |
| v2 | T50, bs3, fb2, 60 ep | best F1 0,876 @K=3 (ep24) | opérating-point réglé |
| v3 (kcurve) | T50, fb2, 60 ep, log K=1..5, sel K3 | **best ep43** | courbes par K |

## Meilleur point (v3, ep43, VALIDATION)

| K | Recall | Précision | F1 | Fausses alarmes |
|---|---|---|---|---|
| 1 | 1,000 | 0,695 | 0,820 | 0,468 |
| 2 | 0,996 | 0,816 | 0,897 | 0,239 |
| 3 | 0,987 | 0,895 | **0,939** | 0,124 |
| 4 | 0,935 | 0,963 | 0,937 | 0,064 |
| 5 | 0,858 | 0,961 | 0,907 | 0,037 |

FrameAcc ~0,74. NB : métriques de détection OSCILLENT epoch-à-epoch (sensibilité de la
règle "≥ K"), mais le modèle sélectionné est solide. Éval TEST : à faire.

## Positionnement (discussion)
L'ordre n'est PAS la clé ici : c'est la SIGNATURE TEMPORELLE du geste (chute rapide vs
assise lente). Le vrai comparateur n'est pas l'ANN mais GRU/LSTM ; l'atout du SNN est
l'EFFICIENCE pour une surveillance continue basse consommation.
