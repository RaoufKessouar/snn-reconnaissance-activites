# MAD — travaux sur caméra événementielle (SNN)

Données : `/data/abdekess61/datasets/lab_gesture_dataset` (événements CSV
`x,y,p,t`, 640 × 480).

Cache de trames : `/data/abdekess61/SResNet/MAD_cache`.

Le parcours maintenu se trouve dans le
[guide de reprise](../../docs/GUIDE_REPRISE.md). Les notes des sous-dossiers
décrivent aussi des expériences historiques ; leurs protocoles de
recalibration peuvent différer du parcours corrigé.

## Structure
- common/        : code partage (mad_io, mad_naming, mad_dataset, mad_chain_dataset, precise_bn)
- 00_extract/    : extraction + inventaire
- 01_visualization/ : stats + planches de frames
- 02_dataset/    : MADDataset + visualisation preprocessed
- 03_train_9class/ : sanity classification 9 activites (plomberie validee)
- 04_train_chain/  : MAD-Chain (experience principale)

## État

- pipeline MAD conservé : recadrage, fenêtre active, trames 128 × 128 ;
- checkpoints MAD-Chain L=3 et L=4 inclus ;
- checkpoint et métriques de détection de chute inclus ;
- scripts d'évaluation corrigés pour ne jamais recalibrer sur le test.

Voir [MAD-Chain](04_train_chain/README.md),
[les résultats](../../docs/RESULTS.md) et
[la reproductibilité](../../docs/REPRODUCIBILITY.md).
