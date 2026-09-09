# Run en cours — tdBN + fenetre active "late"

**Machine :** cetautomatix, GPU 1 · **Batch :** 4 · **Epochs :** 60
**Script :** train_chain_tdbn_aw.py
**Log :** train_chain_tdbn_aw.log
**Checkpoints :** best_chain_tdbn_aw.pth / last_chain_tdbn_aw.pth
**Run W&B :** madchain_TDBN_awLate_L3_T40_bs4_ep60 (projet Article5-MAD-Chain)

## Ce qu'on teste
Sur la meilleure base actuelle (tdBN, test 0,975), on change UNIQUEMENT la
regle de la fenetre active :
- avant : "densite max" (les 3 s les plus denses) -> attrapait la MARCHE
  d'approche (dense) au lieu du geste -> erreurs Sit/Fall pris pour Walk.
- maintenant : "fin d'activite" -> la fenetre se termine a la fin de
  l'activite (dernier mouvement avant immobilite) -> vise le GESTE, exclut
  la marche d'approche situee plus tot.

## Objectif
Voir si les erreurs Sit/Fall -> Walk diminuent grace a un meilleur cadrage
temporel, par-dessus le gain tdBN. Une seule variable change (la fenetre).

## Config
SResNet-38 · neurone LIF · normalisation tdBN · T=40 · seq_len=3 (27 classes)
· window_mode="late" · 70 train / 15 val / 15 test (par participant).

## A regarder
- val(std)/val(ada) : doit rester lisse (comportement tdBN).
- Apres coup : analyse d'erreurs -> les Sit/Fall -> Walk baissent-elles ?
- Comparaison : tdBN "maxdensity" -> test std 0,971 / ada 0,975.
