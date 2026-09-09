# DVS-GC overlap 0/7/8, L=3, T=60

Ce dossier portait initialement le suffixe `T80`, alors que les scripts,
checkpoints et noms de sorties utilisent T=60. Il a été renommé pour éviter
toute ambiguïté.

Configuration :

- primitives : Hand Clapping (0), Arm Roll (7), Air Drums (8) ;
- longueur de chaîne : 3 ;
- classes : 27 ;
- trames : 60 ;
- S-ResNet-38, LIF, BNTT ;
- AdamW, pas 1e-4, décroissance 1e-2.

Points d'entrée :

- `train_overlap.py` : SNN-LIF ;
- `train_ann_bntt_dvsgc.py` : contrôle ANN-BNTT qui démontre la fuite de
  position temporelle ;
- `train_chain_gru_dvsgc.py` : contrôle récurrent.

```bash
export DVSGC_DATA_ROOT=/data/abdekess61/SResNet
WANDB_MODE=offline python train_overlap.py
```

Le résultat 81,96 % de l'étude neurone/reset ne provient pas de ce dossier mais
de `../overlap_078_seq3_T40_neuron_study/`.
