# Run en cours — tdBN + chaines plus longues (seq_len=4)

**Machine :** aida (L40S), GPU 0 · **Batch :** 4 · **Epochs :** 40
**Script :** train_chain_tdbn_L4.py
**Log :** train_chain_tdbn_L4.log
**Checkpoints :** best_chain_tdbn_L4.pth / last_chain_tdbn_L4.pth
**W&B :** hors-ligne (aida sans internet) -> wandb sync a faire apres.

## Ce qu'on teste
La perception d'ordre tient-elle sur des chaines PLUS LONGUES ?
- avant : chaines de 3 activites (27 classes).
- maintenant : chaines de 4 activites (3 primitives -> 81 classes).
On garde les 3 memes primitives (Walk/Sit/Fall) pour isoler la seule
variable "longueur". Classification plate (81 classes, comme l'article).

## Objectif
Verifier que le modele continue de percevoir l'ordre quand la chaine
s'allonge, et tracer la degradation eventuelle (accuracy par position).
C'est la 1re etape avant d'aller vers la detection de chute dans un flux long.

## Config
SResNet-38 · neurone LIF · normalisation tdBN · T=40 · seq_len=4 (81 classes)
· 70 train / 15 val / 15 test (par participant).
Note : une chaine fait toujours 40 frames au total (memoire identique a
seq_len=3), mais 81 classes = ~3x plus de donnees -> ~2 h/epoch.

## A regarder
- accuracy par position + exact-match (toute la chaine correcte).
- R-error (repetition) : l'ordre est-il bien percu ?
- Comparaison : base seq_len=3 (test 0,975, R-error faible).
