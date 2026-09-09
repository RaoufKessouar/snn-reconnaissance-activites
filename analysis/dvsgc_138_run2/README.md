# analysis_output_2

Ce dossier concerne la deuxieme experience principale sur DVS-Gesture-Chain.

## Experience

Modele : SResNet38 ameliore / plus fidele a l'article
Dataset : DVS-Gesture-Chain standard
Gestes primitifs utilises : 1, 3, 8
Longueur des chaines : 4 gestes
Nombre de classes finales : 81
Nombre de frames : 60
Batch size : 4
Gradient accumulation : 2
Batch effectif : 8
Learning rate : 1e-4

## Checkpoint associe

/users/abdekess61/raouf/SResNet/best_model_sresnet38_bs4_accum2_lr1e-4_val.pth

## Resultats principaux

Train accuracy finale : environ 0.9972
Validation accuracy finale : environ 0.9864
Best validation accuracy : environ 0.9957

## Role du dossier

Ce dossier est destine a analyser les resultats de la version SResNet38 :

- predictions sur le test set
- matrice de confusion
- accuracy par classe
- exemples bien classes
- exemples mal classes
- analyse des erreurs restantes
- comparaison avec analysis_output_1

## Remarque

Cette experience correspond au modele principal ameliore sur le DVS-Gesture-Chain standard.
Elle ne correspond pas a l'experience overlap 0/7/8.
