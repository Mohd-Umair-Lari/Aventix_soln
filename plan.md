# KGRAG Pipeline Plan: Real Point Cloud Dataset

## 1. Project Overview & Objectives

Adapt the Knowledge Graph Retrieval-Augmented Generation (KGRAG) pipeline — originally built for synthetic `.ply` data — to the **real SCARLETT optical scanner dataset** produced by the Aventix Combi dispense experiment (`dataset/Point_cloud_dataset/`).

**Goals:**
- Ingest real CSV-based scan files and build per-well point clouds
- Estimate per-well dispensed volumes using empty-plate subtraction
- Construct a Knowledge Graph encoding plates, wells, fill steps, liquids, instruments, and verification outcomes
- Support grounded natural-language Q&A over graph facts via an optional LLM layer

---

## 2. Dataset Summary

| Property | Value |
|---|---|
| Physical plates | 8 (A–H) |
| Filled scans | 15 (W01–W08, D01–D07) |
| Empty reads | 6 (E1–E6; W08 and D05 have no matching empty read) |
| Plate formats | EP384 (384-well, 4.5 mm pitch) · AB0796 (96-well, 9 mm pitch) |
| Liquids | Water (ρ = 0.998 g/mL) · DMSO (ρ = 1.100 g/mL) |
| Fill types | Uniform/cumulative · Ladder (per-column volumes) · Single |
| Ground truth | Plate-level gravimetric only — no per-well reference exists |
| Sensor channels | Two optical sensors per scan (sensor = 1 and sensor = 2) |
| Coordinate origin | A1 nominal centre = (0, 0) mm |

### Experimental Scenarios

| Plate | Liquid | Fill Type | Steps / Volumes |
|---|---|---|---|
| EP384-A | Water | Cumulative | 8.9 → 24.7 → 40.1 µL |
| AB0796-B | Water | Cumulative | 42.6 → 52.3 → 181.8 → 311.1 µL |
| AB0796-C | Water | Ladder | 12 volumes by column (36.5–109 µL) |
| AB0796-D | DMSO | Ladder | 12 volumes by column (14–92.5 µL) |
| AB0796-E | DMSO | Cumulative | 40.1 → 143.7 → 247.2 µL |
| AB0796-F | DMSO | Partial | 5 volumes in column pairs (cols 11–12 empty) |
| AB0796-G | DMSO | Single | 109.3 µL |
| AB0796-H | DMSO | Single | 45.9 µL |

---

## 3. Step 1 — Data Ingestion Layer

**Replace the PLY loader** with a CSV-based ingestion stack.

```
manifest.csv      → experiment master index (one row per filled scan)
wells.csv         → well centres and nominal volumes (per scan × well)
scans/*.csv       → 15 point cloud files (columns: well, row, col, x_mm, y_mm, z_mm, sensor)
empty_reads/*.csv → 6 baseline scans (same column schema, no fluid)
```

- Load `manifest.csv` with `pandas.read_csv`; use it as the loop driver for all downstream processing
- Load `wells.csv` and index by `(scan, well)` for per-well centre lookups
- For each scan in `manifest.csv`, load the corresponding `.csv` from `scans/`
- For each scan, resolve `empty_read` column → load matching file from `empty_reads/`
- **Missing empty reads** (W08, D05): flag with `BASELINE_ESTIMATED` and substitute the mean well-floor Z from other scans of the same plate format

---

## 4. Step 2 — Point Cloud Preprocessing

For each scan and each well:

1. **Well isolation** — extract points whose `(x_mm, y_mm)` falls within the well radius of the nominal centre from `wells.csv`
   - AB0796 (96-well): radius ≈ 4.3 mm
   - EP384 (384-well): radius ≈ 2.0 mm

2. **Dual-sensor fusion** — concatenate `sensor=1` and `sensor=2` point subsets; both channels cover the same well but different height windows. Flag wells where only one sensor returned points (`PARTIAL_SCAN` warning).

3. **Statistical outlier removal** — per-well, reject points beyond `mean ± 3.0 × std_dev` in Z (mirror reference KGRAG approach)

4. **Voxel downsampling** — optional; apply only if point count per well exceeds a configurable threshold (e.g. > 500 points)

---

## 5. Step 3 — Empty-Plate Subtraction *(Key Differentiator)*

This step does not exist in the synthetic reference pipeline and is the foundation of accurate volume estimation.

For each filled scan:

1. Load the matching empty scan (from `manifest.csv → empty_read`)
2. For each well, isolate empty-scan points using the same radius as Step 2
3. Compute `z_floor = median(empty_well_points.z_mm)` — the well-bottom reference height
4. Compute `fluid_height_mm = median(filled_well_points.z_mm) − z_floor`
5. If `fluid_height_mm < 0`: flag as `MEASUREMENT_ANOMALY` (sensor artifact or dry well)
6. If no empty read available: use `z_floor = mean(z_floor values from other plates of same format)`; set status flag `BASELINE_ESTIMATED`

---

## 6. Step 4 — Per-Well Volume Estimation

```
Volume (µL) = fluid_height_mm × π × r²     [cylindrical approximation]
```

| Plate format | Well radius r |
|---|---|
| AB0796 (96-well) | 4.3 mm |
| EP384 (384-well) | 2.0 mm |

**Deviation metrics:**
- vs nominal: `(measured − nominal_uL) / nominal_uL × 100 %`
- vs gravimetric: `(measured − gravimetric_mean_uL_per_well) / gravimetric_mean_uL_per_well × 100 %`

**Special cases:**
- **Ladder fills**: use per-well `nominal_uL` from `wells.csv` (column-specific, not plate-average)
- **Cumulative fills**: each scan step is treated as an independent measurement; volumes should be non-decreasing across steps on the same plate
- **Partial plates** (AB0796-F, cols 11–12 empty): wells with no fluid points → `MISSING_WELL` status

---

## 7. Step 5 — Knowledge Graph Schema

### Node Types

| Node | Key Properties | New vs Reference |
|---|---|---|
| `Lab` | name, location | Existing |
| `Instrument` | name (`SCARLETT`), type (`optical_scanner`) | Existing |
| `Dispenser` | name (`Thermo Multidrop Combi`), channels | Existing |
| `Protocol` | fill_type, liquid, nominal_uL | Existing |
| `Run` | scan_id, scan_start_utc, plate, fill_step | Existing |
| `Plate` | plate_id, plate_type, tare_g, filled_g | Existing |
| `Well` | well_id, row, col, centre_x_mm, centre_y_mm | Existing |
| `Measurement` | fluid_height_mm, estimated_uL, deviation_pct, status | Existing |
| `DispenserChannel` | channel_id (1–8 for 96-well, 1–16 for 384-well) | Existing |
| `PlateFormat` | format_name (EP384 / AB0796), pitch_mm, well_count | **New** |
| `LiquidType` | name (water / DMSO), density_g_per_mL | **New** |
| `FillStep` | step_number, cumulative, nominal_uL | **New** |
| `EmptyRead` | file, scan_start_utc, plate_format | **New** |

### Edge Types

| Edge | From → To | New vs Reference |
|---|---|---|
| `RAN` | Lab → Run | Existing |
| `USED` | Run → Instrument | Existing |
| `FOLLOWED` | Run → Protocol | Existing |
| `HAS_WELL` | Plate → Well | Existing |
| `HAS_MEASUREMENT` | Well → Measurement | Existing |
| `MEASURED_BY` | Measurement → Instrument | Existing |
| `FILLED_BY` | Well → DispenserChannel | Existing |
| `SUPERSEDED_BY` | Measurement → Measurement | Existing |
| `HAS_FORMAT` | Plate → PlateFormat | **New** |
| `USED_LIQUID` | Run → LiquidType | **New** |
| `HAS_STEP` | Run → FillStep | **New** |
| `FOLLOWS_STEP` | FillStep → FillStep | **New** (cumulative fills) |
| `BASELINE_FROM` | Measurement → EmptyRead | **New** |
| `HAS_EMPTY_READ` | Plate → EmptyRead | **New** |

---

## 8. Step 6 — Deterministic Verification Rules

All rules run before any LLM layer. Status values: `PASS` · `FAIL` · `WARNING` · `UNVERIFIED`.

| Rule | Condition | Status |
|---|---|---|
| RULE_1 | `\|deviation_pct vs gravimetric\| > tolerance` (default 15 %) | FAIL |
| RULE_2 | Systematic bias across all wells of one dispenser channel | WARNING CHANNEL_BIAS |
| RULE_3 | Point count < threshold **or** `std_dev_mm` > threshold | WARNING BUBBLE_OR_MISREAD |
| RULE_4 *(new)* | Only one sensor returned data for a well | WARNING PARTIAL_SCAN |
| RULE_5 | Calibration record missing or expired | UNVERIFIED |
| RULE_6 *(new)* | No matching empty read; fallback Z-floor used | WARNING BASELINE_ESTIMATED |
| RULE_7 *(new)* | `fluid_height_mm > well_depth_mm` | FAIL OVERFILL |
| RULE_8 *(new)* | Cumulative step volume < previous step volume for same well | FAIL STEP_REGRESSION |
| RULE_9 *(new)* | Well expected to be filled but zero fluid points found | FAIL MISSING_WELL |

---

## 9. Step 7 — Multi-Step & Ladder Fill Logic

### Cumulative Fills (plates A, B, E)

- Each scan step creates a new `Measurement` node linked by `FOLLOWS_STEP`
- Cross-step consistency check: `V[step_n] ≥ V[step_n−1]` per well; violation → `FAIL STEP_REGRESSION`
- Final step `Measurement` is the canonical measurement; earlier steps linked via `SUPERSEDED_BY`

### Ladder Fills (plates C, D)

- Per-well `nominal_uL` comes from `wells.csv` (column-specific)
- Build a column-volume map; expected profile is monotonically increasing across columns
- Validate with Spearman rank correlation; flag non-monotonic columns

### Partial Plate (plate F)

- Columns 11–12 are intentionally empty → expected `MISSING_WELL` status (not an anomaly)
- Encode expected-empty wells in `Protocol` node as `expected_empty_cols: [11, 12]`

---

## 10. Step 8 — Grounded RAG Query Layer

Adapt the `ask()` function from the reference notebook to support:

| Query type | Example |
|---|---|
| Single well | `"What volume was measured in well B3 of plate AB0796-B?"` |
| Plate summary | `"Summarise all measurements for plate AB0796-E."` |
| Liquid filter | `"Which DMSO plates passed verification at step 1?"` |
| Fill step | `"Compare step 2 vs step 3 for plate EP384-A."` |
| Cross-plate | `"Which plate had the highest well-to-well CV?"` |
| Anomaly query | `"List all wells that triggered a CHANNEL_BIAS warning."` |

**LLM integration (optional):**
- Provider: Groq `llama-3.3-70b-versatile` via `GROQ_API_KEY` env variable
- LLM receives only verified graph facts as context; it narrates, never invents
- If `GROQ_API_KEY` is absent, the system returns structured graph output directly

---

## 11. Step 9 — Visualisation

| Output | Description |
|---|---|
| Deviation heatmap | 8×12 (96-well) or 16×24 (384-well) grid coloured by `deviation_pct` (`RdYlGn` colormap) |
| Fill-step curve | Volume vs fill_step per well (scatter + line); one plot per cumulative plate |
| Ladder profile | Mean well volume per column vs setpoint; error bars = well-to-well std dev |
| KG topology | NetworkX / matplotlib graph of nodes and edges for a single plate run |
| Sensor comparison | Side-by-side histogram of Z values from sensor 1 vs sensor 2 per well |

---

## 12. Validation & Testing

| Assertion | Scope |
|---|---|
| Gravimetric vs estimated deviation < 15 % | All uniform and single-fill scans |
| Ladder column ordering is monotonic | Plates C and D |
| Cumulative fill volumes are non-decreasing per well | Plates A, B, E |
| All status codes covered (PASS / FAIL / WARNING / UNVERIFIED) | Full dataset |
| Wells with no empty read are flagged `BASELINE_ESTIMATED` | W08, D05 |
| Partial-plate empty wells correctly show `MISSING_WELL` | Plate F, cols 11–12 |
| Dual-sensor fusion produces ≥ reference point count | All wells |

---

## 13. File & Module Structure

```
3D spatial DL/
├── plan.md                          ← this file
├── kgrag_wellplate.ipynb            ← reference pipeline (synthetic PLY)
├── aventix_kgrag_real.ipynb         ← new notebook (real CSV dataset)
├── dataset/
│   └── Point_cloud_dataset/         ← in .gitignore
│       └── Point cloud dataset - Combi dispense 2026-11/
│           ├── manifest.csv
│           ├── wells.csv
│           ├── scans/
│           └── empty_reads/
└── synthetic_dataset/               ← in .gitignore
```

**New notebook section order** (mirroring reference KGRAG structure):

1. Setup & Config
2. Dataset Audit & CSV Loader
3. Point Cloud Preprocessing & Sensor Fusion
4. Empty-Plate Subtraction
5. Per-Well Volume Estimation
6. Deterministic Verification Engine
7. Knowledge Graph Construction
8. Multi-Step & Ladder Fill Logic
9. Grounded RAG Query Layer
10. Visualisation
11. Validation Assertions

---

## 14. Dependencies

```python
# Core
pandas
numpy
scipy          # KDTree, stats
networkx       # Knowledge Graph
matplotlib
seaborn        # Heatmaps

# Point Cloud
open3d         # Retained for visualisation utilities

# Optional LLM
groq           # pip install groq
python-dotenv  # for GROQ_API_KEY loading
```

---

> [!IMPORTANT]
> Per-well ground truth does not exist. All verification is relative to the **plate-level gravimetric mean**. Per-well `PASS/FAIL` is a statistical inference, not a measured fact.

> [!NOTE]
> Volumes above ~28 µL were dispensed in multiple shots by the Combi dispenser. This introduces within-well spatial non-uniformity that may require Z-surface fitting (plane or polynomial) rather than simple median Z for accurate height estimation in future iterations.

> [!TIP]
> Start notebook development with plate **AB0796-G** (single fill, 109.3 µL, one scan, one empty read) — the simplest scenario — before tackling cumulative or ladder fills.

