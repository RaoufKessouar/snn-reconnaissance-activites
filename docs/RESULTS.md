# Résultats et niveau de vérification

Ce document sépare trois niveaux de preuve :

- **trace retrouvée** : la valeur apparaît dans un journal d'exécution conservé ;
- **rapport** : la valeur figure dans le rapport final mais sa trace exacte n'a
  pas été retrouvée dans le dossier remis ;
- **à reproduire** : le code et le checkpoint existent, mais l'évaluation doit
  être relancée pour obtenir une preuve autonome.

Les journaux bruts ne sont pas publiés car leurs métadonnées contiennent des
informations personnelles et des chemins internes. Les valeurs vérifiées sont
résumées ici avec leur contexte.

## DVS-Gesture-Chain

### Reproduction standard 1/3/8

| Configuration | Split | Score | Preuve |
|---|---:|---:|---|
| S-ResNet-38, L=4, T=60 | test | 97,48 % | rapport, à reproduire avec le checkpoint principal |
| ancien `best_model.pth` | test | 86,88 % | trace retrouvée |

La seule trace de test conservée charge explicitement l'ancien
`best_model.pth`, et non
`best_model_sresnet38_bs4_accum2_lr1e-4_val.pth`. Elle ne réfute donc pas
nécessairement 97,48 %, mais elle ne le prouve pas non plus. Le script
`experiments/dvsgc_standard/test.py` charge désormais le checkpoint principal
par défaut pour lever cette ambiguïté lors de la prochaine exécution.

### Primitives à recouvrement spatial 0/7/8

| Modèle | L | T | Split | Score | Preuve |
|---|---:|---:|---|---:|---|
| ANN-ReLU + BN | 3 | 40 | validation | 34,64 % | trace retrouvée |
| SNN-IF, reset zéro | 3 | 40 | validation | 72,50 % | trace retrouvée |
| SNN-LIF, reset soustractif | 3 | 40 | validation | 81,96 % | trace retrouvée |
| ANN-ReLU + BNTT | 3 | 60 | validation | 98,57 % | trace retrouvée |

La dernière ligne démontre que BNTT peut coder la position temporelle même
quand le neurone n'a pas de mémoire. Elle ne constitue donc pas un contrôle
valide de la dynamique.

La configuration et le journal du résultat 81,96 % utilisent **T=40**. Le
rapport associe cette valeur à T=60 dans la section 3.6 et le tableau 1. Le dépôt
retient T=40 comme valeur prouvée tant qu'un journal T=60 avec le même protocole
de recalibration n'est pas produit.

## MAD-Chain

Les participants sont séparés strictement :

- entraînement : 1 à 70 ;
- validation : 71 à 85 ;
- test : 86 à 100.

| Modèle | L | Normalisation | Split et taille | Score | Preuve |
|---|---:|---|---|---:|---|
| ANN sans mémoire | 3 | BN | test, 1 215 | 8,81 % | trace retrouvée |
| ANN sans mémoire | 3 | tdBN | test, 405 | 31,85 % | rapport, à reproduire |
| SNN-LIF | 3 | BNTT | test, 1 215 | 78,44 % | rapport |
| SNN-LIF | 3 | tdBN | test, 1 215 | 97,12 % | trace retrouvée, mode standard |
| SNN-LIF | 4 | tdBN | test, 1 215 | 96,93 % | trace retrouvée, mode standard |
| encodeur CNN + GRU | 3 | sans objet | test, 405 | 100,00 % | trace retrouvée |

Le rapport indique 99,93 % pour le GRU, alors que la trace remise indique
100,00 %. Avec 405 décisions de séquence, 99,93 % n'est pas un score possible
par simple exactitude (une erreur donnerait 99,75 %). La valeur 100,00 % est
donc conservée comme valeur vérifiable, sans conclure sur l'origine du 99,93 %.

Les checkpoints tdBN historiques ont été sélectionnés après une mise à jour de
leurs statistiques sur les entrées de validation. Les tests « standard »
ci-dessus n'utilisent pas les entrées du test pour adapter le modèle, mais le
checkpoint avait déjà vu la distribution de validation. Les scripts corrigés
recalculent désormais ces statistiques uniquement sur l'entraînement. Une
réplication stricte reste nécessaire pour confirmer les scores à l'identique.

## Détection de chute

Le checkpoint est sélectionné sur le F1 de validation à K=3. Le test contient
450 chaînes, dont 232 positives.

| K | Sensibilité | Précision | F1 | Fausses alarmes |
|---:|---:|---:|---:|---:|
| 1 | 0,996 | 0,724 | 0,838 | 0,404 |
| 2 | 0,996 | 0,837 | 0,909 | 0,206 |
| **3** | **0,991** | **0,902** | **0,945** | **0,115** |
| 4 | 0,957 | 0,937 | 0,947 | 0,069 |
| 5 | 0,892 | 0,963 | 0,926 | 0,037 |

Ces valeurs sont celles du rapport. Le journal exact du test n'a pas été
retrouvé dans le dossier remis ; le checkpoint et
`eval_fall_split.py` permettent de les recalculer. K=3 reste le seuil retenu
car il a été choisi sur la validation, même si K=4 produit a posteriori un F1
test légèrement supérieur.

## Ce qui ne doit pas être comparé directement

- Les scores DVS-GC et MAD-Chain n'utilisent ni les mêmes gestes ni les mêmes
  participants.
- Les contrôles ANN-tdBN et GRU utilisent 405 chaînes de test, contre 1 215 pour
  le SNN principal.
- Le GRU possède un encodeur différent de S-ResNet-38.
- Les expériences historiques n'ont utilisé qu'une graine ; elles ne donnent
  pas d'intervalle de confiance.
