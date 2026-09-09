# Phase 2a — MAD : dataset et pipeline d'extraction

## Dataset MAD (ETIS, Béranger 2025)
Multimodal (radar, caméra événementielle, MoCap — on exploite l'event-camera, sensor=1),
anonymisant. 100 participants (genre/âge équilibrés). 9 activités : Walk(1), Stand Up(2),
Sit Down(3), Up Stairs(4), Down Stairs(5), Pick an Object(6), Step Over(7), Semi Turn(8),
Fall(9), déclinées en directions (29 variantes). 3 répétitions, séquences 10 s.
Caméra Prophesee EVK1-VGA 640x480, résolution temporelle ~µs. CSV, 1 ligne/événement.

## Pipeline (code : experiments/mad/common/)
1. `mad_naming.parse` : regex A{act}{sub}P{part}R{rep}S{sensor}D{ant}.
2. `mad_io.load_events` : lecture CSV ROBUSTE (pd.to_numeric coerce + dropna) — corrige un
   cas réel de ligne corrompue qui plantait np.bincount.
3. `active_window(t, W_us=3s)` : fenêtre glissante à DENSITÉ MAX d'événements
   (bincount 50 ms → cumsum → argmax).
4. `make_frames` : crop (120,520,80,440) → sous-échantillonnage 128x128, 2 canaux polarité,
   accumulation en F images (fi = (t-ws)*F/(we-ws)).
5. `MADChainDataset` : classes = produit cartésien des primitives (27 à L=3) ; répartition
   MULTINOMIALE des T=40 images entre segments (durées variables ; entrée toujours
   [B,40,2,128,128]) ; cache .npz avec tag de config ; augment = miroir horizontal (x).

## Split
PAR PARTICIPANT : train 1-70 / val 71-85 / test 86-100 → sujets de test JAMAIS vus
(exigence plus forte que DVS-GC).
