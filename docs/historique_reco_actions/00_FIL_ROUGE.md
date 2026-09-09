# Fil rouge — Reconnaissance d'actions par SNN sur données événementielles

**Question directrice.** Les SNN sont censés apporter un avantage sur les données
événementielles grâce à leur dynamique temporelle interne. Les tâches sur lesquelles
on les évalue mobilisent-elles réellement cette dynamique ? Et cela tient-il sur des
activités réelles et des sujets inédits ?

## Chronologie (4 phases)

| Phase | Sujet | Fichier détail | Résultat clé |
|---|---|---|---|
| 0 | DVS-Gesture : le benchmark insuffisant | `01_dvs_gesture.md` | résolu sans temporel (ANN 97 %) |
| 1 | DVS-GC (synthétique, ordre) | `02_dvsgc_overlap.md` | SNN 0,82 vs ANN 0,35 ; instabilité BNTT diagnostiquée |
| 2 | MAD + MAD-Chain (réel) | `03_mad_pipeline.md`, `04_madchain_diagnostic.md`, `05_tdbn.md`, `06_experiences_complementaires.md` | tdBN : 0,78 → 0,975 ; ANN 0,088 ; L=4 scale |
| 3 | Détection de chute (application) | `07_detection_chute.md` | F1 0,94 (K=3), recall 0,99 |

Tableau de tous les chiffres : `99_RESULTATS.md`.

## Récit condensé

1. **DVS-Gesture** se résout à ~97 % par un classifieur d'images SANS mémoire temporelle
   → le benchmark ne mesure pas l'atout des SNN. (détail : `01_dvs_gesture.md`)

2. **DVS-GC** rend l'ordre constitutif de la classe. Sur la campagne {0,7,8} (27 classes) :
   le SNN atteint 0,82, l'ANN de contrôle plafonne (train 0,41, val 0,35) → incapacité
   STRUCTURELLE (somme temporelle invariante par permutation). On y diagnostique aussi
   l'instabilité de validation = calibration de BNTT (stats sur peu d'échantillons,
   activations parcimonieuses ; pire à T grand). (détail : `02_dvsgc_overlap.md`)

3. **MAD-Chain** transpose le protocole à des activités réelles (Walk/Sit/Fall), split
   PAR SUJET. Baseline BNTT = 0,78 mais validation oscillante. Diagnostic par élimination
   (SWA, cosine LR, T=50, T=30, flip — tous écartés) → instabilité INTRINSÈQUE à BNTT.
   Remède ciblé : **tdBN** (stats sur T×B + calage seuil) → 0,78 → 0,975, oscillation
   disparue. Contrôle ANN sur réel : 0,088 (proche du hasard). Chaînes L=4 (81 classes) :
   l'ordre PASSE À L'ÉCHELLE (0,969). Deux pistes tranchées : fenêtre « late » (écartée),
   étude neurone QIF/ALIF (en cours).
   (détails : `03_`, `04_`, `05_`, `06_`)

4. **Détection de chute** : reformulation détection+localisation dans un flux, lecture
   par image, règle « ≥ K images ». Meilleur point : recall 0,99, F1 0,94, fausses
   alarmes 0,12 (K=3). (détail : `07_detection_chute.md`)

## Méthodologie (fil conducteur transversal)
Vérité terrain d'abord (lire le code, pas le handoff) · une variable à la fois ·
diagnostiquer AVANT de ré-entraîner · accepter que la mesure contredise l'hypothèse
(fenêtre active, cosine LR) · les résultats négatifs sont des résultats.
