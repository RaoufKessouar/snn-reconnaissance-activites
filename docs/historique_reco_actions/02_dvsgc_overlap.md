# Phase 1 — DVS-GC, campagne overlap {0,7,8}, L=3, 27 classes

Primitives : Hand Clapping (0), Arm Roll (7), Air Drums (8) — choisies pour leurs
zones d'activité RECOUVRANTES (tâche volontairement dure : pas de raccourci spatial).
Modèle : SResNet-38, LIF (fuite 0,874, tau≈8, reset par soustraction, decay_input=False),
BNTT (60 BN, momentum 0,1, eps 1e-4), AdamW (lr 1e-4, wd 0,01), BPTT + gradient de
substitution. Hasard = 0,037.

## Entraînements et résultats

| Run | Config | Résultat | Note |
|---|---|---|---|
| T=60 | BNTT, bs4+accum2, 50 ep | acc propre 0,7679 ; val OSCILLE (0,53↔0,71) | best-checkpoint non fiable |
| T=40 | BNTT | val plus lisse | résolution retenue |
| Run A | LIF+Sub, T40, bs6, precise-BN, 80 ep | **best val 0,8196** | oscillation ÷3, mesure fiabilisée |
| ANN-BN | ReLU + BN, somme temporelle | train PLAFONNE 0,41 · val 0,3464 | contrôle |
| SNN | (Run A) | train ~0,99 · val ~0,82 | |

## Contrôle ANN vs SNN — lecture
Le train de l'ANN PLAFONNE à 0,41 : il n'apprend même pas le train (capacité pourtant
suffisante — le SNN atteint 0,99). Ce n'est pas du surapprentissage mais une IMPOSSIBILITÉ
STRUCTURELLE : la somme sur le temps est invariante par permutation → (A,B) et (B,A)
indiscernables. → la dynamique temporelle du SNN est NÉCESSAIRE.

## Stabilité de la mesure (precise-BN)
À T grand, chaque image reçoit moins d'événements (activité DILUÉE) → stats BNTT plus
bruitées → validation oscille. Recalculer les stats BNTT sur le train avant chaque éval
(precise-BN) → sélection fiable, val ~0,82. NB : ne traite que la MESURE, pas la cause.

## Métriques d'analyse (réutilisées ensuite sur MAD)
- Exactitude par position : 0,861 / 0,845 / 0,827 (léger déclin ~2 pts/pos, amplitude 3,4 pts).
- R-error (sur chaînes fausses, ≥1 position fausse == précédent) : R_pred(==pred[p-1])=0,4538
  (> 0,333, répète sa propre sortie) · R_true(==vrai[p-1])=0,1846 (< 0,333, ne persévère pas
  sur l'entrée). Cohérent avec le reset par soustraction.
- Confusion : 0→[474,38,28] acc0,878 · 7→[44,460,96] acc0,767 · 8→[26,30,484] acc0,896.
  Confusion dominante Arm Roll → Air Drums (96) = la difficulté voulue.

## Diagnostic BNTT (hypothèses, méthode par élimination)
- H1 instabilité = calibration BNTT : eval() (running stats) acc 0,9957 vs train() (batch
  stats bs4) 0,6475 → running stats bonnes, batch stats (petit B) mauvaises ; l'instabilité
  vient de running stats NON CONVERGÉES pendant l'entraînement. CONFIRMÉE (direction corrigée).
- H2 dataset stochastique : cache .npz + seed → val set figé. RÉFUTÉE.
- H3 T grand ↑ instabilité via parcimonie. PLAUSIBLE (T60 pire que T40).
- H4 plafond = confusion spatiale 7↔8. CONFIRMÉE.
- H5 erreur de répétition (reset soustraction). PRÉSENTE mais SECONDAIRE.

## Piste évoquée
Run B (IF + reset à zéro, l'ensemble IF+Zero — pas LIF+Zero) pour attaquer la répétition
(article Table 4). Non menée jusqu'au bout dans notre travail.

## Verdict de phase
Démonstration établie mais sur benchmark SYNTHÉTIQUE (gestes labo, peu de sujets).
→ transposer au réel = phase 2.
