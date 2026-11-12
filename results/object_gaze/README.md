# Object gaze: agent vs. Atari-HEAD humans (Asterix)

Stand 2026-11-12, Branch `feat/object-gaze-metrics`.

## Daten

| Ordner/Datei | Inhalt | Erzeugt mit |
|---|---|---|
| `agent/seed{0,1,2}_step*.csv` | 10 Episoden pro Seed, Modell `grid5x5_square24x24_EMMA_foveated_effort`, Cluster-Job 3310213 | `scripts/vision_grid/eval_objects.sh` |
| `human/full_trials.csv` | 15 Trials (je 15 min, 1 Episode) | `scripts/human_object_gaze.py` |
| `human/max_frames_5000.csv` | erste 5000 Frames pro Trial | `... --max-frames 5000` |
| `human/max_score_7000.csv`, `max_score_12000.csv` | bis Score 7000 bzw. 12000 | `... --max-score N` |
| `human/play_stats_*.csv` | Avatar-Positionen und Blickverhalten (gleiche Schnitte) | `scripts/human_play_stats.py` |
| `comparison.csv` | Objekt-Anteile pro Gruppe (Mensch-Schnitt, Agent-Seed, alle Seeds) und Kategorie: Mittel, SD, Min, Max, Zufall, Verhältnis, im Bild | `scripts/object_gaze_report.py` |
| `gaze_heatmaps.png`, `gaze_grid.csv` | Blickzeit pro Ort: Menschen pixelgenau und auf dem Grid (Mittel über Trials), Agent auf dem Grid, sobald `agent/*play_stats*.csv` vorliegen | `scripts/object_gaze_report.py` |

Methode: `embodied/core/object_gaze.py` (Objekt-Anteile) und `embodied/core/play_stats.py`
(Avatar/Blick). Alles ist zeitgewichtet (Agent: Emulator-Frames à 50 ms, Mensch: duration(ms)
pro Frame) und auf dem Vision-Grid des Agenten gemessen (5x5 Quadrate à 40x52 px).
Zufallsniveau = Anteil der Grid-Quadrate, die eine Kategorie überlappen.

## 1. Anteil der Zeit auf Objektkategorien (Grid)

In % der Episodenzeit, in Klammern das Zufallsniveau, × = Verhältnis zum Zufall
(Mittel der Verhältnisse pro Episode/Trial).

| Kategorie | Mensch erste 5000 Frames | Mensch ganzer Trial | Agent Ø 3 Seeds | Agent Seed 1 (bester) |
|---|---|---|---|---|
| player | 57,5 (9,4) · 6,1× | 60,1 (10,4) · 5,8× | 15,9 (9,3) · 1,7× | 20,1 (9,9) · 2,1× |
| enemy | 28,8 (13,6) · 2,1× | 32,3 (17,5) · 1,8× | 14,4 (11,2) · 1,3× | 15,9 (11,4) · 1,4× |
| item | 30,0 (7,8) · 3,9× | 37,5 (9,6) · 3,9× | 17,6 (12,3) · 1,4× | 19,3 (12,1) · 1,6× |
| reward | 45,8 (21,8) · 2,1× | 44,1 (19,5) · 2,3× | 24,7 (16,7) · 1,5× | 27,0 (15,8) · 1,7× |

- Spielfortschritt ist vergleichbar: Menschen haben nach 5000 Frames 6520 ± 880 Punkte, der Agent
  ~6800 Punkte in ~5000 Frames (Seed 1: 11440 in 6460 Frames). Der Agent stirbt nur früher
  (alle Agent-Episoden enden durch Game Over nach 1,2–7,5 min).
- Schnitte bei Score 7000/12000 ändern die Menschenwerte kaum (player 57,5/57,4 %).
  Das Blickverhalten der Menschen ist über die Levels stabil, der Unterschied ist kein Level-Effekt.

### Nur das beste Modell (Seed 1, Step 37,7 M)

Mensch geschnitten bei Score 12000 (≈ gleicher Spielabschnitt wie Seed 1), Spannweite über Trials
bzw. Episoden, 95 %-Bootstrap-KI für Seed 1:

| Kategorie | Mensch (Score ≤ 12000) | Seed 1 Mittel [95 %-KI] | Seed 1 Spannweite |
|---|---|---|---|
| player | 57,4 [51,7–64,2] | 20,1 [18,4–21,9] | 14,7–25,9 |
| enemy | 31,7 [19,3–38,0] | 15,9 [14,6–17,1] | 11,4–18,6 |
| item | 31,4 [28,3–34,4] | 19,3 [17,3–21,0] | 12,2–24,1 |
| reward | 41,9 [36,0–46,7] | 27,0 [24,6–29,7] | 20,8–36,7 |

- Bei player und item überschneiden sich die Spannweiten nicht: die beste Agent-Episode liegt unter dem
  schlechtesten Menschen-Trial. Dafür reichen 10 Episoden.
- Innerhalb des Agenten hängt der Blick auf den Avatar stark mit dem Score zusammen
  (r = 0,80 über alle 30 Episoden, r = 0,90 innerhalb Seed 1). Enemy/item/reward: r = 0,2–0,3.
  Aber: In den besten Episoden steigt auch das Zufallsniveau für player (bis 13,5 %).

## 2. Avatar-Positionen und Blickverhalten (Menschen)

Agent-Werte folgen aus dem neuen Eval-Lauf (`agent_play_stats.csv`). Lokaler Test des Skripts mit
einem fast untrainierten Checkpoint: Blick-Entropie 4,5 bit (gleichverteilt), 4 Wechsel/s.

| Größe | Mensch erste 5000 Frames | Mensch ganzer Trial |
|---|---|---|
| Avatar gefunden | 99,6 % | 98,3 % |
| Avatar x Mittel ± SD (px, Bild 160) | 76,3 ± 15,8 | 76,1 ± 18,0 |
| Spurwechsel pro min | 66 | 80 |
| horizontale Geschwindigkeit (px/s) | 13,0 | 18,5 |
| Zeit pro Spur 0–7 (oben→unten, %) | 5 · 6 · 11 · 28 · 27 · 10 · 8 · 5 | 6 · 8 · 12 · 23 · 24 · 12 · 9 · 6 |
| Blick-Entropie (bit, max 4,64) | 3,44 | 3,45 |
| Quadratwechsel pro s | 2,9 | 2,8 |
| mittlere Verweildauer (ms) | 353 | 365 |
| Blick im Avatar-Quadrat | 34,5 % | 34,7 % |
| Blick ≤ 1 Quadrat vom Avatar | 84,6 % | 87,5 % |
| Abstand Quadratmitte–Avatar (px) | 31,9 | 31,1 |
| Zeile Blick − Avatar (−2…+2, %) | 0,8 · 14,8 · 56,0 · 23,2 · 3,4 | 1,1 · 16,0 · 56,7 · 21,7 · 3,0 |
| Spalte Blick − Avatar (−2…+2, %) | 4,6 · 14,3 · 60,1 · 15,7 · 4,9 | 3,3 · 16,3 · 59,1 · 17,2 · 3,7 |

Blickzeit pro Grid-Quadrat, Menschen, erste 5000 Frames (%):

```
 0.1   1.6   1.1   0.5   0.1
 2.1   7.3  14.5   7.7   1.4
 2.5   8.3  26.3  10.6   2.9
 0.4   2.6   3.9   2.5   0.6
 0.0   0.2   2.5   0.4   0.1
```

Menschen halten den Avatar in der Bildmitte (Spuren 3–4, x ≈ 76) und schauen fast immer auf ihn oder
ein Nachbarquadrat, eher darunter als darüber.

Hinweise zum Vergleich:
- Der Agent kann nur einmal pro Env-Schritt wechseln (≈ 5 Frames, ≥ ~250 ms). Menschen werden pro Frame
  (50 ms) gezählt, mit dem häufigsten Quadrat des Frames.
- Agent-Avatar aus RAM, Menschen-Avatar aus dem Farbdetektor. Beide stimmen bei Zufallsspiel überein
  (x ±1 px, y in 99,5 % exakt).
- Beim Agenten wird pro Schritt das gewählte Quadrat gezählt, die alten Quadrate während EMMA-Sakkaden
  bleiben hier unberücksichtigt (anders als in Teil 1).
