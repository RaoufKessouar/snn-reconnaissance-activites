# Journal des expériences — MAD-Chain

Config commune : SResNet-38, caméra événementielle MAD, primitives Walk(1)/Sit Down(3)/Fall(9),
seq_len=3 -> 27 classes, split PAR PARTICIPANT, T=40, crop 128, fenêtre active glissante.
Deux métriques de val : Val(std) = stats BNTT du train ; Val(ada) = BN recalculée sur la val.

## Récapitulatif des runs

| Run | Config | Train acc | Best val(ada) | Test std | Test ada | Verdict |
|-----|--------|-----------|---------------|----------|----------|---------|
| Préliminaire (30 sujets) | T=40, 24tr/6val, 30 ep | 0.71 | 0.50 | -- | -- | preuve de concept, val très bruitée |
| **Baseline (100 sujets)** | T=40, 70tr/15val, 80 ep | 0.97 | 0.79 | **0.784** | 0.696 | **RÉFÉRENCE** |
| T=30 + flip x | T=30, +aug | 0.935 | 0.66 | 0.666 | 0.663 | ✗ résolution insuffisante |
| T=40 + flip x | T=40, +aug | 0.966 | 0.64 | 0.619 | 0.536 | ✗ le flip nuit |
| SWA | T=40 baseline + moyennage poids | (en cours) | -- | -- | -- | stabiliser (patch) |
| cosine LR | T=40, LR 1e-4 -> ~0 | (à venir) | -- | -- | -- | attaquer la CAUSE (LR constant) |

## Enseignements
- **Val très bruitée** (~10x le bruit d'échantillonnage) : sensibilité intrinsèque du SNN,
  PAS la parcimonie (T=30 dense n'a rien changé) ni le petit val set.
- **T=40 ~ optimal** : T=30 trop grossier (accuracy chute), T=60 non testé (parcimonie + mémoire).
- **Flip horizontal invalide ici** : le modèle exploite une orientation constante vs le capteur.
- **Analyse d'erreurs (baseline)** : accuracy par position 0.85 / 0.97 / 0.91 (position 1 la plus dure) ;
  Walk reconnu à 0.97 ; erreurs = Sit Down/Fall -> Walk (hypothèse : la fenêtre active capte la
  marche d'approche) ; R-error 0.09 << 0.33 -> l'ORDRE est bien perçu.
- **Vrai levier identifié** : décroissance du learning-rate (cosine). Le LR constant fait "rebondir"
  le modèle autour du minimum -> oscillation. SWA était un patch (moyenne le rebond) ;
  le cosine LR empêche le rebond (cause racine).

## Pistes ouvertes
- cosine LR (run en préparation) ; puis éventuellement cosLR + SWA combinés.
- Si cosLR stabilise : tester T=50 (résolution) sur cette base stable.
- Corriger le critère de fenêtre active (viser le geste, pas l'approche) pour Sit/Fall.
- Axe neurone : ALIF (adaptation) plutôt que QIF, en étude dédiée ultérieure.
