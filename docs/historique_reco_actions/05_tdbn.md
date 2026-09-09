# Phase 2c — tdBN : le résultat principal

## Principe
BNTT estime ses stats PAR PAS DE TEMPS (sur B valeurs). tdBN (Zheng 2021) les estime
CONJOINTEMENT sur T×B, avec un calage sur le seuil :
  x_hat = alpha * V_th * (x - mu)/sqrt(sigma^2 + eps) ;  y = gamma*x_hat + beta
- mu, sigma sur T×B ; gamma, beta partagés (l'indice t disparaît).
- alpha*V_th cale l'échelle sur le seuil de décharge → fraction saine de neurones spike
  à chaque couche (débloque l'apprentissage en profondeur, pas que la stabilité).

## Coût d'implémentation
Forward PAS-À-PAS → COUCHE-PAR-COUCHE (conv sur tous les T → tdBN sur T → LIF déroulé).
Mémoire ∝ T×B (~38 Go à T=40, bs=4). Vraie manip, pas un hyperparamètre.
Fichiers : block_tdbn.py, model_tdbn.py, train_chain_tdbn.py.

## Résultats (27 classes, test = sujets 86-100)

| Normalisation | Val (ada) | Test (std) | Test (ada) | Oscillation |
|---|---|---|---|---|
| BNTT (baseline) | 0,79 | 0,78 | 0,78 | forte (±0,6) |
| BNTT + SWA | 0,85 | 0,665 | 0,731 | masquée |
| **tdBN** | **0,969** | **0,971** | **0,975** | disparue (±0,05) |

**+19 points** en test. Oscillation disparue, std ≈ ada.

## Crédibilité du résultat
1. Triple accord val 0,969 ≈ test std 0,971 ≈ test ada 0,975 (ensembles disjoints).
2. std n'utilise QUE les stats d'entraînement → pas de coup de pouce transductif.
3. ANN reste à 0,088 sur la même tâche → tâche non triviale.
- val ≥ train expliqué : augmentation train-only ; train = moyenne mobile sur modèle qui
  progresse vs val = instantané final ; en éval, running stats mieux calibrées.

## Portée (nuance honnête)
Les auteurs CONNAISSAIENT tdBN (cité) mais ont choisi BNTT EXPRÈS : les paramètres par
pas de temps sont leur OBJET D'ÉTUDE (attention temporelle). tdBN est le bon choix pour
NOTRE régime (activités transitoires, sujets inédits, peu d'exemples/classe), pas pour le
leur. Contribution = ADAPTER la normalisation au régime, pas corriger une faute.
