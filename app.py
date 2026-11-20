from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
candidate_dirs = [
    Path(os.getenv("AVENTIX_DATASET_DIR", ROOT / "dataset" / "Point_cloud_dataset")),
    ROOT.parent / "dataset" / "Point_cloud_dataset",
]
DATASET_DIR = candidate_dirs[0]
for c in candidate_dirs:
    if (c / "manifest.csv").exists():
        DATASET_DIR = c
        break
    nested = list(c.glob("*/manifest.csv")) if c.is_dir() else []
    if nested:
        DATASET_DIR = nested[0].parent
        break

CONFIG = {
    "TOLERANCE_PCT": 15.0,
    "OUTLIER_SIGMA": 3.0,
    "MIN_POINTS": 10,
    "CALIBRATION_ID": "SCARLETT-REAL-UNVERIFIED",
}
SURFACE_TENSION = {"water": 0.0728, "dmso": 0.0436}
DENSITY_MAP = {"water": 0.998, "dmso": 1.100}
CONTACT_ANGLE_DEG = {"water": 60.0, "dmso": 45.0}


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: json_value(v) for k, v in row.items()} for row in frame.to_dict("records")]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    use_llm: bool = True


class Pipeline:
    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.manifest = pd.read_csv(dataset_dir / "manifest.csv")
        self.wells = pd.read_csv(dataset_dir / "wells.csv")
        self.graph = nx.MultiDiGraph()
        self.measurements = self._measure()
        self._verify()
        self._build_graph()

    @staticmethod
    def _clean(points: pd.DataFrame) -> pd.DataFrame:
        if len(points) < 3: return points.copy()
        z = points.z_mm.astype(float)
        lo = z.mean() - CONFIG["OUTLIER_SIGMA"] * z.std()
        hi = z.mean() + CONFIG["OUTLIER_SIGMA"] * z.std()
        return points[(z >= lo) & (z <= hi)].copy()

    @staticmethod
    def _isolate(points: pd.DataFrame, x: float, y: float, radius: float) -> pd.DataFrame:
        return points[((points.x_mm - x) ** 2 + (points.y_mm - y) ** 2) <= radius ** 2].copy()

    @staticmethod
    def _meniscus(liquid: str, density: float, radius: float) -> tuple[float, float]:
        liq = str(liquid).lower()
        gamma = SURFACE_TENSION.get(liq, 0.070)
        rho = float(density) * 1000.0 if pd.notna(density) else DENSITY_MAP.get(liq, 1.0) * 1000.0
        theta_deg = CONTACT_ANGLE_DEG.get(liq, 50.0)
        cos_theta = np.cos(np.radians(theta_deg))
        h = min((2.0 * gamma * cos_theta / (rho * 9.81 * radius * 1e-3)) * 1e3, radius * 0.75)
        return h, 0.5 * np.pi * radius ** 2 * h

    def _volume(self, height: float, plate_type: str, liquid: str, density: float) -> float:
        if not np.isfinite(height) or height <= 0: return np.nan
        r0, k = (0.85, 0.11) if "384" in str(plate_type) else (1.65, 0.21)
        _, vm = self._meniscus(liquid, density, r0 + k * height)
        return float((np.pi / (3.0 * k)) * ((r0 + k * height) ** 3 - r0 ** 3) + 0.3 * vm)

    @staticmethod
    def _surface(points: pd.DataFrame, ptype: str) -> float:
        rim = 10.0 if ptype == "EP384" else 13.2
        s2 = points[(points.sensor == 2) & (points.z_mm > 5.0) & (points.z_mm < rim)]
        if len(s2) >= 10: return float(s2.z_mm.median())
        s1 = points[(points.sensor == 1) & (points.z_mm > 1.8) & (points.z_mm < rim)]
        if len(s1) >= 5: return float(s1.z_mm.median())
        valid = points[points.z_mm < rim]
        return float(valid.z_mm.median()) if len(valid) else np.nan

    @staticmethod
    def _floor(points: pd.DataFrame, ptype: str) -> float:
        thresh = 2.0 if ptype == "EP384" else 1.5
        floor = points[(points.sensor == 1) & (points.z_mm < thresh)]
        if len(floor) >= 5: return float(floor.z_mm.median())
        valid = points[points.z_mm < (thresh + 1.0)]
        return float(valid.z_mm.median()) if len(valid) else (1.25 if ptype == "EP384" else 0.80)

    def _measure(self) -> pd.DataFrame:
        empty_cache = {str(x): pd.read_csv(self.dataset_dir / str(x)) for x in self.manifest.empty_read.dropna().unique()}
        rows = []
        for _, meta in self.manifest.iterrows():
            scan_id = Path(meta.file).stem
            filled = pd.read_csv(self.dataset_dir / meta.file)
            ptype = "EP384" if "384" in str(meta.plate_type) else "AB0796"
            radius = 1.5 if ptype == "EP384" else 3.0
            empty = empty_cache.get(str(meta.empty_read)) if pd.notna(meta.empty_read) else None
            scan_wells = self.wells[self.wells.scan == scan_id]
            for _, wr in scan_wells.iterrows():
                fp = self._clean(self._isolate(filled, wr.centre_x_mm, wr.centre_y_mm, radius))
                zs = self._surface(fp, ptype)
                zf = self._floor(self._clean(self._isolate(empty, wr.centre_x_mm, wr.centre_y_mm, radius)), ptype) if empty is not None else (1.25 if ptype == "EP384" else 0.80)
                height = zs - zf if np.isfinite(zs) and np.isfinite(zf) else np.nan
                volume = self._volume(height, ptype, str(meta.liquid).lower(), meta.density_g_per_mL)
                hm, vm = self._meniscus(str(meta.liquid).lower(), meta.density_g_per_mL, radius) if np.isfinite(height) else (np.nan, np.nan)
                sensors = set(fp.sensor.astype(int)) if len(fp) else set()
                well_row = ord(wr.well[0].upper()) - 64
                well_col = int(re.sub(r"[^0-9]", "", wr.well))
                chan_num = well_row
                max_r = 16 if ptype == "EP384" else 8
                max_c = 24 if ptype == "EP384" else 12
                is_edge = (well_row == 1) or (well_row == max_r) or (well_col == 1) or (well_col == max_c)

                rows.append({
                    **meta.to_dict(), "scan_id": scan_id, "well": wr.well,
                    "row": well_row, "col": well_col, "channel": chan_num, "is_edge": is_edge,
                    "centre_x_mm": wr.centre_x_mm, "centre_y_mm": wr.centre_y_mm, "radius_mm": radius,
                    "point_count": len(fp), "sensor_count": len(sensors), "z_floor_mm": zf, "z_surface_mm": zs,
                    "fluid_height_mm": height, "meniscus_height_mm": hm, "meniscus_volume_uL": vm,
                    "estimated_uL": volume, "nominal_uL": wr.nominal_uL, "baseline_estimated": empty is None
                })
        out = pd.DataFrame(rows)
        out["deviation_pct"] = np.where(
            out.nominal_uL > 0,
            (out.estimated_uL - out.nominal_uL) / out.nominal_uL * 100,
            np.where(out.estimated_uL.isna(), 0.0, out.estimated_uL)
        )
        out["gravimetric_deviation_pct"] = np.where(
            pd.notna(out.gravimetric_mean_uL_per_well) & (out.gravimetric_mean_uL_per_well > 0),
            (out.estimated_uL - out.gravimetric_mean_uL_per_well) / out.gravimetric_mean_uL_per_well * 100,
            out.deviation_pct
        )
        return out

    def _verify(self):
        df = self.measurements
        df["rule_id"] = ""
        df["fault_reason"] = ""
        df["status"] = "PASS"

        for idx, r in df.iterrows():
            reasons = []
            ptype = "EP384" if "384" in str(r.plate_type) else "AB0796"
            max_depth = 10.5 if ptype == "EP384" else 14.5

            if pd.isna(r.estimated_uL) or r.point_count == 0:
                reasons.append(("RULE_9", "MISSING_WELL", "FAIL"))
            if pd.notna(r.fluid_height_mm) and r.fluid_height_mm > max_depth:
                reasons.append(("RULE_7", "OVERFILL", "FAIL"))
            if pd.notna(r.gravimetric_deviation_pct) and abs(r.gravimetric_deviation_pct) > CONFIG["TOLERANCE_PCT"]:
                reasons.append(("RULE_1", "GRAVIMETRIC_DEVIATION", "FAIL"))
            if r.point_count < CONFIG["MIN_POINTS"]:
                reasons.append(("RULE_3", "BUBBLE_OR_MISREAD", "WARNING"))
            if r.sensor_count < 2 and r.point_count > 0:
                reasons.append(("RULE_4", "PARTIAL_SCAN", "WARNING"))
            if r.baseline_estimated:
                reasons.append(("RULE_6", "BASELINE_ESTIMATED", "WARNING"))
            if r.is_edge and pd.notna(r.gravimetric_deviation_pct) and abs(r.gravimetric_deviation_pct) > CONFIG["TOLERANCE_PCT"]:
                reasons.append(("RULE_5", "EDGE_EFFECT", "WARNING"))

            if reasons:
                df.loc[idx, "fault_reason"] = "; ".join(x[1] for x in reasons)
                df.loc[idx, "rule_id"] = ";".join(x[0] for x in reasons)
                df.loc[idx, "status"] = "FAIL" if any(x[2] == "FAIL" for x in reasons) else "WARNING"

        for p in ["EP384-A", "AB0796-B", "AB0796-E"]:
            p_wells = df[df.plate == p].sort_values(["well", "fill_step"])
            for _, w_grp in p_wells.groupby("well"):
                vols = w_grp.estimated_uL.values
                indices = w_grp.index.values
                for i in range(1, len(vols)):
                    if pd.notna(vols[i]) and pd.notna(vols[i-1]) and (vols[i] < vols[i-1] - 1.0):
                        cur_idx = indices[i]
                        df.loc[cur_idx, "rule_id"] = (df.loc[cur_idx, "rule_id"] + ";RULE_8").strip(";")
                        df.loc[cur_idx, "fault_reason"] = (df.loc[cur_idx, "fault_reason"] + "; STEP_REGRESSION").strip(";")
                        df.loc[cur_idx, "status"] = "FAIL"

        for scan_id, s_grp in df.groupby("scan_id"):
            ptype = "EP384" if "384" in str(s_grp.plate_type.iloc[0]) else "AB0796"
            max_c = 24 if ptype == "EP384" else 12
            valid = s_grp[s_grp.nominal_uL > 0]
            if valid.empty: continue

            interior = valid[~valid.is_edge]
            scan_baseline = interior["gravimetric_deviation_pct"].median() if len(interior) >= 10 else valid["gravimetric_deviation_pct"].median()

            for ch, ch_grp in valid.groupby("channel"):
                ch_test = ch_grp[(ch_grp.col > 1) & (ch_grp.col < max_c)]
                if len(ch_test) < 4: ch_test = ch_grp
                rel_devs = ch_test["gravimetric_deviation_pct"] - scan_baseline
                bad = rel_devs[rel_devs.abs() > 10.0]
                if len(bad) >= 4 and np.sign(bad).nunique() == 1 and abs(rel_devs.median()) > 10.0:
                    for b_idx in bad.index:
                        cur_rules = df.loc[b_idx, "rule_id"]
                        if "RULE_2" not in cur_rules:
                            df.loc[b_idx, "rule_id"] = (cur_rules + ";RULE_2").strip(";")
                            df.loc[b_idx, "fault_reason"] = (df.loc[b_idx, "fault_reason"] + "; CHANNEL_BIAS").strip(";")

    def _build_graph(self):
        g = self.graph
        g.add_node("Lab:Aventix", type="Lab", name="Aventix Precision Lab")
        g.add_node("Instrument:SCARLETT", type="Instrument", instrument_type="optical_scanner", manufacturer="Aventix")
        g.add_node("Instrument:Combi", type="Instrument", instrument_type="liquid_dispenser", model="Thermo Multidrop Combi")

        for liquid, grp in self.measurements.groupby("liquid"):
            g.add_node(f"Liquid:{liquid}", type="Liquid", name=str(liquid),
                       density_g_per_mL=float(grp.density_g_per_mL.dropna().iloc[0]),
                       surface_tension_N_m=SURFACE_TENSION.get(str(liquid).lower(), 0.070))

        for plate, grp in self.measurements.groupby("plate"):
            first = grp.iloc[0]
            pn = f"Plate:{plate}"
            ptype = "EP384" if "384" in str(first.plate_type) else "AB0796"
            g.add_node(pn, type="Plate", plate_id=plate, plate_type=ptype, wells_count=int(first.wells))
            if pd.notna(first.empty_read):
                eid = f"EmptyRead:{Path(str(first.empty_read)).stem}"
                g.add_node(eid, type="EmptyRead", file=str(first.empty_read))
                g.add_edge(pn, eid, relation="HAS_BASELINE_SCAN")

        for scan, grp in self.measurements.groupby("scan_id"):
            first = grp.iloc[0]
            rn = f"Run:{scan}"
            pn = f"Plate:{first.plate}"
            ln = f"Liquid:{first.liquid}"
            ptype = "EP384" if "384" in str(first.plate_type) else "AB0796"
            layout_type = "ladder" if "ladder" in str(first.layout).lower() else "uniform"
            fp_text = f"{ptype}|{first.liquid}|{layout_type}|{CONFIG['CALIBRATION_ID']}"
            fingerprint = hashlib.sha256(fp_text.encode("utf-8")).hexdigest()[:12]

            g.add_node(rn, type="Run", scan_id=scan, file=first.file, fill_step=int(first.fill_step),
                       plate_type=ptype, liquid=str(first.liquid), layout=layout_type,
                       fingerprint=fingerprint, scan_start_utc=str(first.scan_start_utc))
            for a, b, rel in [("Lab:Aventix", rn, "CONDUCTED_RUN"), (rn, pn, "TARGETED_PLATE"),
                              (rn, ln, "USED_LIQUID"), (rn, "Instrument:Combi", "DISPENSED_BY")]:
                g.add_edge(a, b, relation=rel)

        wells_added = set()
        for _, r in self.measurements.iterrows():
            pn = f"Plate:{r.plate}"
            wn = f"Well:{r.plate}:{r.well}"
            mn = f"Measurement:{r.plate}:{r.scan_id}:{r.well}"
            cn = f"DispenserChannel:{r.channel}"

            if wn not in wells_added:
                wells_added.add(wn)
                g.add_node(wn, type="Well", plate_id=r.plate, well_id=r.well, row=int(r.row), col=int(r.col),
                           centre_x_mm=float(r.centre_x_mm), centre_y_mm=float(r.centre_y_mm))
                g.add_edge(pn, wn, relation="HAS_WELL")

            g.add_node(mn, type="Measurement", measurement_id=mn, scan_id=r.scan_id, well_id=r.well, plate_id=r.plate,
                       source_file=r.file, estimated_uL=json_value(r.estimated_uL), nominal_uL=json_value(r.nominal_uL),
                       deviation_pct=json_value(r.deviation_pct), gravimetric_deviation_pct=json_value(r.gravimetric_deviation_pct),
                       status=r.status, fault_reason=r.fault_reason, rule_id=r.get("rule_id", ""),
                       channel=int(r.channel), fluid_height_mm=json_value(r.fluid_height_mm),
                       point_count=int(r.point_count), sensor_count=int(r.sensor_count),
                       baseline_estimated=bool(r.baseline_estimated))
            g.add_edge(wn, mn, relation="HAS_MEASUREMENT")
            g.add_edge(mn, "Instrument:SCARLETT", relation="MEASURED_BY")

            g.add_node(cn, type="DispenserChannel", channel_number=int(r.channel), instrument="Instrument:Combi")
            g.add_edge(mn, cn, relation="DISPENSED_BY_CHANNEL")

    def _resolve_run_node(self, run_key: str) -> str | None:
        g = self.graph
        if run_key in g: return run_key
        with_prefix = f"Run:{run_key}"
        if with_prefix in g: return with_prefix
        cand = [n for n, d in g.nodes(data=True) if d.get("type") == "Run" and (n.lower().startswith(f"run:{run_key.lower()}") or str(d.get("scan_id", "")).lower().startswith(run_key.lower()))]
        return cand[0] if cand else None

    def are_runs_comparable(self, run_a: str, run_b: str) -> tuple[bool, str]:
        g = self.graph
        na = self._resolve_run_node(run_a)
        nb = self._resolve_run_node(run_b)
        if not na or not nb:
            return False, "One or both runs not found in graph."
        fp_a = g.nodes[na].get("fingerprint")
        fp_b = g.nodes[nb].get("fingerprint")
        if fp_a == fp_b:
            return True, f"Matching CDE fingerprint [{fp_a}]: runs share plate format, liquid, and protocol."
        diffs = []
        for k in ["plate_type", "liquid", "layout"]:
            va = g.nodes[na].get(k)
            vb = g.nodes[nb].get(k)
            if va != vb: diffs.append(f"{k} differs ('{va}' vs '{vb}')")
        return False, f"Incompatible configurations (fingerprint {fp_a} vs {fp_b}): {'; '.join(diffs)}."

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        q = question.lower()
        plates = {p for p in self.manifest.plate.unique() if p.lower() in q}
        for m in re.findall(r"\bplate\s*[-_]?\s*([a-h]|ep384[-_]?a?)\b", q):
            plates.update({p for p in self.manifest.plate.unique() if (p.endswith(m.upper()) if not m.startswith("ep") else p == "EP384-A")})
        wells = [x.upper() for x in re.findall(r"\b([a-p]\d{1,2})\b", q)]
        target_channels = [int(c) for c in re.findall(r"\bchannel\s*(\d+)\b", q)]

        hits = []
        candidate_plates = plates if plates else self.manifest.plate.unique()
        for p_id in candidate_plates:
            pn = f"Plate:{p_id}"
            if pn not in self.graph: continue

            for _, wn, edata in self.graph.out_edges(pn, data=True):
                if edata.get("relation") != "HAS_WELL": continue
                w_attrs = self.graph.nodes[wn]
                w_id = w_attrs.get("well_id")
                if wells and w_id not in wells: continue

                for _, mn, m_edata in self.graph.out_edges(wn, data=True):
                    if m_edata.get("relation") != "HAS_MEASUREMENT": continue
                    m_attrs = self.graph.nodes[mn]

                    ch_num = None
                    for _, cn, c_edata in self.graph.out_edges(mn, data=True):
                        if c_edata.get("relation") == "DISPENSED_BY_CHANNEL":
                            ch_num = self.graph.nodes[cn].get("channel_number")

                    if target_channels and ch_num not in target_channels: continue
                    if any(t in q for t in ["failed", "fault", "anomal", "issue", "bias"]) and m_attrs.get("status") not in {"FAIL", "WARNING", "UNVERIFIED"}:
                        continue

                    hits.append({"measurement_id": mn, "well_id": w_id, "plate_id": p_id, "channel": ch_num, **m_attrs})

        return hits[:50]

    @staticmethod
    def _verify_grounding(response_text: str, facts: list[dict[str, Any]]) -> tuple[bool, str]:
        fact_ids = {f["measurement_id"] for f in facts}
        cited_ids = re.findall(r"Measurement:[A-Za-z0-9_\-\.:]+", response_text)
        for cid in cited_ids:
            if not any(cid in fid or fid in cid for fid in fact_ids):
                return False, f"LLM fabricated Measurement ID: {cid}"

        cited_decimals = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", response_text)]
        ground_truth = []
        for f in facts:
            for k in ["estimated_uL", "nominal_uL", "deviation_pct", "gravimetric_deviation_pct"]:
                val = f.get(k)
                if val is not None and np.isfinite(val): ground_truth.append(float(val))

        for c_val in cited_decimals:
            matches = any(abs(c_val - g_val) < 0.2 or abs(abs(c_val) - abs(g_val)) < 0.2 for g_val in ground_truth)
            if not matches and c_val > 1.0:
                meta_match = any(str(round(c_val, 1)) in f["measurement_id"] for f in facts)
                if not meta_match:
                    return False, f"LLM cited unverified numeric value: {c_val}"

        return True, "All cited figures and IDs verified against graph facts."

    def answer(self, question: str, use_llm: bool = True) -> dict[str, Any]:
        q = question.lower()
        if "comparable" in q or "compare" in q:
            runs_found = [Path(f).stem for f in self.manifest.file if Path(f).stem.lower() in q]
            plates_found = [p for p in self.manifest.plate.unique() if p.lower() in q]
            if len(runs_found) >= 2:
                comp, reason = self.are_runs_comparable(runs_found[0], runs_found[1])
                return {"answer": f"CDE Comparability Assessment: {comp}. {reason}", "evidence_pack": [], "grounding_verified": True}
            elif len(plates_found) >= 2:
                run_a = self.measurements[self.measurements.plate == plates_found[0]].scan_id.iloc[0]
                run_b = self.measurements[self.measurements.plate == plates_found[1]].scan_id.iloc[0]
                comp, reason = self.are_runs_comparable(run_a, run_b)
                return {"answer": f"CDE Comparability Assessment for {plates_found[0]} vs {plates_found[1]}: {comp}. {reason}", "evidence_pack": [], "grounding_verified": True}

        facts = self.retrieve(question)
        if not facts:
            return {"answer": "No verified data found in the graph for this query.", "evidence_pack": [], "grounding_verified": True}

        if use_llm and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                context = "\n".join(
                    f"- {f['measurement_id']}: Well {f['well_id']} on Plate {f['plate_id']} (Scan {f['scan_id']}). Status={f['status']}, Est={f.get('estimated_uL', np.nan):.2f} uL, Nom={f.get('nominal_uL', np.nan):.2f} uL, GravDev={f.get('gravimetric_deviation_pct', 0.0):.1f}%. Rules=[{f.get('rule_id', 'none')}]. Fault={f.get('fault_reason', 'none')}."
                    for f in facts[:20]
                )
                prompt = f"Answer in 1-3 concise natural language sentences using only these graph facts. Cite Measurement IDs. Do not invent causes or numbers.\nQuestion: {question}\nFacts:\n{context}"
                client = Groq(api_key=os.environ["GROQ_API_KEY"])
                raw_ans = client.chat.completions.create(
                    model=os.getenv("MODEL", "openai/gpt-oss-120b"),
                    temperature=0.1,
                    messages=[{"role": "system", "content": "You are a grounded laboratory QC consultant."},
                              {"role": "user", "content": prompt}]
                ).choices[0].message.content.strip()

                is_grounded, msg = self._verify_grounding(raw_ans, facts)
                if not is_grounded:
                    raw_ans = f"[Grounding Warning: {msg}]\n" + raw_ans

                return {"answer": raw_ans, "evidence_pack": facts, "grounding_verified": is_grounded}
            except Exception:
                pass

        lines = [f"Grounded Findings ({len(facts)} measurement{'s' if len(facts) != 1 else ''} retrieved via graph traversal):"]
        for f in facts[:10]:
            lines.append(f"Well {f['well_id']} ({f['plate_id']}, Scan {f['scan_id']}): Status {f['status']} | Measured: {f['estimated_uL']:.2f} uL (Nominal: {f['nominal_uL']:.2f} uL, GravDev: {f.get('gravimetric_deviation_pct', 0.0):+.1f}%) | Rules: [{f.get('rule_id') or 'None'}] | Reasons: {f.get('fault_reason') or 'None'}")
        if len(facts) > 10:
            lines.append(f"... and {len(facts) - 10} more measurements.")
        return {"answer": "\n".join(lines), "evidence_pack": facts, "grounding_verified": True}


app = FastAPI(title="Aventix Scarlett KGRAG API", version="1.0.0")
pipeline: Pipeline | None = None


@app.on_event("startup")
def startup():
    global pipeline
    pipeline = Pipeline()


def service() -> Pipeline:
    if pipeline is None: raise HTTPException(503, "Pipeline is not initialized")
    return pipeline


@app.get("/health")
def health():
    p = service()
    return {"status": "ok", "dataset": str(p.dataset_dir), "scans": len(p.manifest),
            "measurements": len(p.measurements), "graph_nodes": p.graph.number_of_nodes(),
            "graph_edges": p.graph.number_of_edges()}


@app.get("/measurements")
def get_measurements(plate: str | None = None, scan_id: str | None = None, status: str | None = None,
                     well: str | None = None, limit: int = Query(100, ge=1, le=5000)):
    p = service()
    frame = p.measurements
    for col, val in [("plate", plate), ("scan_id", scan_id), ("status", status), ("well", well)]:
        if val: frame = frame[frame[col].astype(str).str.upper() == val.upper()]
    return {"count": len(frame), "items": records(frame.head(limit))}


@app.get("/summary")
def summary(plate: str | None = None):
    p = service()
    frame = p.measurements if not plate else p.measurements[p.measurements.plate.str.upper() == plate.upper()]
    if frame.empty: raise HTTPException(404, "Plate not found")
    agg = frame.groupby("plate").agg(
        scans=("scan_id", "nunique"),
        wells=("well", "count"),
        mean_uL=("estimated_uL", "mean"),
        mean_abs_dev_pct=("gravimetric_deviation_pct", lambda x: x.abs().mean())
    ).reset_index()
    return {"items": records(agg)}


@app.get("/comparability")
def comparability(run_a: str, run_b: str):
    p = service()
    same, reason = p.are_runs_comparable(run_a, run_b)
    return {"comparable": same, "reason": reason}


@app.post("/query")
def query(request: QueryRequest):
    return service().answer(request.question, request.use_llm)


@app.get("/graph")
def graph(node_type: str | None = None, limit: int = Query(1000, ge=1, le=10000)):
    p = service()
    nodes = [{"id": n, **{k: json_value(v) for k, v in d.items()}} for n, d in p.graph.nodes(data=True) if not node_type or d.get("type") == node_type]
    selected = {x["id"] for x in nodes[:limit]}
    edges = [{"source": a, "target": b, **{k: json_value(v) for k, v in d.items()}} for a, b, d in p.graph.edges(data=True) if a in selected and b in selected]
    return {"nodes": nodes[:limit], "edges": edges}
