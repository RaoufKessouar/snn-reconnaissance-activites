# analysis_output_1

Ce dossier concerne la premiere experience complete sur DVS-Gesture-Chain.

## Experience

Modele : premiere version SResNet/SResNest implementee
Dataset : DVS-Gesture-Chain standard
Gestes primitifs utilises : 1, 3, 8
Longueur des chaines : 4 gestes
Nombre de classes finales : 81
Nombre de frames : 60
Batch size : 4

## Resultats principaux

Train accuracy finale : environ 0.9319
Validation accuracy : environ 0.8636
Test accuracy : environ 0.8688
Test loss : environ 0.6401

## Role du dossier

Ce dossier contient les premieres analyses du modele :

- predictions sur le test set
- matrice de confusion
- accuracy par classe
- exemples bien classes
- exemples mal classes
- analyse temporelle par position dans la chaine
- confusions entre gestes primitifs

## Remarque

Cette experience sert de baseline initiale avant la version SResNet38 plus fidele.
