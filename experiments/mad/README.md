# MAD — travaux camera evenementielle (SNN)

Donnees : /data/abdekess61/datasets/lab_gesture_dataset (CSV event x,y,p,t, 640x480)
Cache frames : /data/abdekess61/SResNet/MAD_cache

## Structure
- common/        : code partage (mad_io, mad_naming, mad_dataset, mad_chain_dataset, precise_bn)
- 00_extract/    : extraction + inventaire
- 01_visualization/ : stats + planches de frames
- 02_dataset/    : MADDataset + visualisation preprocessed
- 03_train_9class/ : sanity classification 9 activites (plomberie validee)
- 04_train_chain/  : MAD-Chain (experience principale)

## Etat
- Pipeline MAD valide (crop 128, fenetre active glissante, T=40).
- Experience en cours : MAD-Chain (voir 04_train_chain/README.md).
