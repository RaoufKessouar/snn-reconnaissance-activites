# Experience overlap_078_seq3_T40

Objectif : construire une variante DVS-Gesture-Chain plus ambigue spatialement.

Classes primitives utilisees :
- 0 : Hand Clapping
- 7 : Arm Roll
- 8 : Air Drums

Motivation :
Ces gestes activent des zones proches du haut du corps et des bras. L'objectif est de reduire les raccourcis purement spatiaux et de tester si le modele exploite mieux la dynamique temporelle.

Configuration :
- T = 40 frames
- seq_len = 3 gestes par chaine
- class_num = 3
- nombre de classes finales = 3^3 = 27
- architecture = SResNet38
- batch_size = 8
- epochs = 40
- lr = 1e-4

Important :
La copie dvsgc_overlap.py modifie aussi le dossier de cache genere :
DVSGC_overlap_078_seq3_cls3_frames_40_split_by_number

Cela evite de reutiliser l'ancien dataset DVSGC_frames_number_60_split_by_number.
