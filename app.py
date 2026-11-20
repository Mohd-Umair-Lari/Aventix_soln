from __future__ import annotations

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
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("AVENTIX_DATASET_DIR", ROOT / "dataset" / "Point_cloud_dataset"))
CONFIG = {"TOLERANCE_PCT": 15.0, "OUTLIER_SIGMA": 3.0, "MIN_POINTS": 10,
          "CALIBRATION_ID": "SCARLETT-REAL-UNVERIFIED"}
SURFACE_TENSION = {"water": 0.0728, "dmso": 0.0436}
DENSITY_MAP = {"water": 0.998, "dmso": 1.100}


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
        gamma = SURFACE_TENSION.get(str(liquid).lower(), 0.070)
        rho = float(density) * 1000.0 if pd.notna(density) else DENSITY_MAP.get(str(liquid).lower(), 1.0) * 1000.0
        h = min((2.0 * gamma / (rho * 9.81 * radius * 1e-3)) * 1e3, radius * 0.75)
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
        floor = points[(points.sensor == 1) & (points.z_mm < 3.0)]
        if len(floor) >= 5: return float(floor.z_mm.median())
        valid = points[points.z_mm < 4.0]
        return float(valid.z_mm.median()) if len(valid) else (1.25 if ptype == "EP384" else 0.82)

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
                zf = self._floor(self._clean(self._isolate(empty, wr.centre_x_mm, wr.centre_y_mm, radius)), ptype) if empty is not None else (1.25 if ptype == "EP384" else 0.82)
                height = zs - zf if np.isfinite(zs) and np.isfinite(zf) else np.nan
                volume = self._volume(height, ptype, str(meta.liquid).lower(), meta.density_g_per_mL)
                hm, vm = self._meniscus(str(meta.liquid).lower(), meta.density_g_per_mL, radius) if np.isfinite(height) else (np.nan, np.nan)
                sensors = set(fp.sensor.astype(int)) if len(fp) else set()
                reasons = []
                if len(fp) == 0: reasons.append("MISSING_WELL")
                if empty is None: reasons.append("BASELINE_ESTIMATED")
                if len(sensors) < 2 and len(fp) > 0: reasons.append("PARTIAL_SCAN")
                rows.append({**meta.to_dict(), "scan_id": scan_id, "well": wr.well,
                    "row": ord(wr.well[0].upper()) - 64, "col": int(re.sub(r"[^0-9]", "", wr.well)),
                    "centre_x_mm": wr.centre_x_mm, "centre_y_mm": wr.centre_y_mm, "radius_mm": radius,
                    "point_count": len(fp), "sensor_count": len(sensors), "z_floor_mm": zf, "z_surface_mm": zs,
                    "fluid_height_mm": height, "meniscus_height_mm": hm, "meniscus_volume_uL": vm,
                    "estimated_uL": volume, "nominal_uL": wr.nominal_uL, "baseline_estimated": empty is None,
                    "status": "FAIL" if not len(fp) else ("WARNING" if empty is None else "PASS"),
                    "fault_reason": ";".join(reasons)})
        out = pd.DataFrame(rows)
        out["deviation_pct"] = (out.estimated_uL - out.nominal_uL) / out.nominal_uL * 100
        out["gravimetric_deviation_pct"] = (out.estimated_uL - out.gravimetric_mean_uL_per_well) / out.gravimetric_mean_uL_per_well * 100
        return out

    def _verify(self):
        for i, r in self.measurements.iterrows():
            reasons = []
            if pd.isna(r.estimated_uL): reasons.append(("RULE_9", "MISSING_WELL"))
            elif abs(r.gravimetric_deviation_pct) > CONFIG["TOLERANCE_PCT"]: reasons.append(("RULE_1", "GRAVIMETRIC_DEVIATION"))
            if r.point_count < CONFIG["MIN_POINTS"]: reasons.append(("RULE_3", "BUBBLE_OR_MISREAD"))
            if r.sensor_count < 2 and r.point_count > 0: reasons.append(("RULE_4", "PARTIAL_SCAN"))
            if r.baseline_estimated: reasons.append(("RULE_6", "BASELINE_ESTIMATED"))
            unique = list(dict.fromkeys(name for _, name in reasons))
            self.measurements.loc[i, "fault_reason"] = "; ".join(unique)
            self.measurements.loc[i, "rule_id"] = ";".join(rule for rule, _ in reasons if _ in unique)
            self.measurements.loc[i, "status"] = "FAIL" if any(rule in {"RULE_1", "RULE_9"} for rule, _ in reasons) else ("WARNING" if reasons else "PASS")

    def _build_graph(self):
        g = self.graph
        g.add_node("Lab:Aventix", type="Lab", name="Aventix Precision Lab")
        g.add_node("Instrument:SCARLETT", type="Instrument", instrument_type="optical_scanner", manufacturer="Aventix")
        g.add_node("Instrument:Combi", type="Instrument", instrument_type="liquid_dispenser", model="Thermo Multidrop Combi")
        for liquid, grp in self.measurements.groupby("liquid"):
            g.add_node(f"Liquid:{liquid}", type="Liquid", name=str(liquid), density_g_per_mL=float(grp.density_g_per_mL.dropna().iloc[0]), surface_tension_N_m=SURFACE_TENSION.get(str(liquid).lower(), .070))
        for plate, grp in self.measurements.groupby("plate"):
            first = grp.iloc[0]; pn = f"Plate:{plate}"
            g.add_node(pn, type="Plate", plate_id=plate, plate_type="EP384" if "384" in str(first.plate_type) else "AB0796", wells_count=int(first.wells))
            if pd.notna(first.empty_read): g.add_node(f"EmptyRead:{Path(str(first.empty_read)).stem}", type="EmptyRead", file=str(first.empty_read)); g.add_edge(pn, f"EmptyRead:{Path(str(first.empty_read)).stem}", relation="HAS_BASELINE_SCAN")
        for scan, grp in self.measurements.groupby("scan_id"):
            first = grp.iloc[0]; rn = f"Run:{scan}"; pn = f"Plate:{first.plate}"; ln = f"Liquid:{first.liquid}"
            g.add_node(rn, type="Run", scan_id=scan, file=first.file, fill_step=int(first.fill_step), layout=str(first.layout), scan_start_utc=str(first.scan_start_utc))
            for a, b, rel in [("Lab:Aventix", rn, "CONDUCTED_RUN"), (rn, pn, "TARGETED_PLATE"), (rn, ln, "USED_LIQUID"), (rn, "Instrument:Combi", "DISPENSED_BY")]: g.add_edge(a, b, relation=rel)
        for _, r in self.measurements.iterrows():
            wn=f"Well:{r.plate}:{r.well}"; mn=f"Measurement:{r.plate}:{r.scan_id}:{r.well}"
            g.add_node(wn, type="Well", plate_id=r.plate, well_id=r.well, row=int(r.row), col=int(r.col), centre_x_mm=float(r.centre_x_mm), centre_y_mm=float(r.centre_y_mm))
            g.add_node(mn, type="Measurement", measurement_id=mn, scan_id=r.scan_id, well_id=r.well, plate_id=r.plate, source_file=r.file, estimated_uL=json_value(r.estimated_uL), nominal_uL=json_value(r.nominal_uL), deviation_pct=json_value(r.gravimetric_deviation_pct if pd.notna(r.gravimetric_deviation_pct) else r.deviation_pct), status=r.status, fault_reason=r.fault_reason, rule_id=r.get("rule_id", ""), point_count=int(r.point_count), sensor_count=int(r.sensor_count), baseline_estimated=bool(r.baseline_estimated))
            g.add_edge(wn, mn, relation="HAS_MEASUREMENT"); g.add_edge(mn, "Instrument:SCARLETT", relation="MEASURED_BY")

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        q = question.lower(); plates = {p for p in self.manifest.plate.unique() if p.lower() in q}
        for m in re.findall(r"\bplate\s*[-_]?\s*([a-h]|ep384[-_]?a?)\b", q):
            plates.update({p for p in self.manifest.plate.unique() if (p.endswith(m.upper()) if not m.startswith("ep") else p == "EP384-A")})
        wells = [x.upper() for x in re.findall(r"\b([a-p]\d{1,2})\b", q)]
        hits=[]
        for n,d in self.graph.nodes(data=True):
            if d.get("type") != "Measurement" or (plates and d.get("plate_id") not in plates) or (wells and d.get("well_id") not in wells): continue
            if any(t in q for t in ["failed", "fault", "anomal", "issue"]) and d.get("status") not in {"FAIL", "WARNING", "UNVERIFIED"}: continue
            hits.append({"measurement_id": n, **d})
        return hits[:50]

    def answer(self, question: str, use_llm: bool) -> dict[str, Any]:
        facts = self.retrieve(question)
        if not facts: return {"answer": "No verified data found in the graph for this query.", "evidence_pack": []}
        if use_llm and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                context = "\n".join(f"- {f['measurement_id']}: Well {f['well_id']} on Plate {f['plate_id']} (Scan {f['scan_id']}). Status={f['status']}, Est={f['estimated_uL']:.2f} uL, Nom={f['nominal_uL']:.2f} uL, Dev={f['deviation_pct']:.1f}%. Fault={f['fault_reason'] or 'none'}." for f in facts[:20])
                prompt = f"Answer in 1-3 concise sentences using only these graph facts. Cite Measurement IDs. Do not invent causes or numbers.\nQuestion: {question}\nFacts:\n{context}"
                client=Groq(api_key=os.environ["GROQ_API_KEY"]); response=client.chat.completions.create(model=os.getenv("MODEL", "openai/gpt-oss-120b"), temperature=.1, messages=[{"role":"system","content":"You are a grounded laboratory QC consultant."},{"role":"user","content":prompt}])
                return {"answer": response.choices[0].message.content.strip(), "evidence_pack": facts}
            except Exception:
                pass
        lines=[f"Grounded Findings ({len(facts)} measurement{'s' if len(facts)!=1 else ''} retrieved):"]
        for f in facts[:10]: lines.append(f"Well {f['well_id']} ({f['plate_id']}, Scan {f['scan_id']}): Status {f['status']} | Measured: {f['estimated_uL']:.2f} uL (Nominal: {f['nominal_uL']:.2f} uL, Dev: {f['deviation_pct']:+.1f}%) | Rules: [{f.get('rule_id') or 'None'}] | Reasons: {f.get('fault_reason') or 'None'}")
        return {"answer":"\n".join(lines), "evidence_pack":facts}


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
    p=service(); return {"status":"ok", "dataset":str(p.dataset_dir), "scans":len(p.manifest), "measurements":len(p.measurements), "graph_nodes":p.graph.number_of_nodes(), "graph_edges":p.graph.number_of_edges()}


@app.get("/measurements")
def measurements(plate: str|None=None, scan_id: str|None=None, status: str|None=None, well: str|None=None, limit: int=Query(100, ge=1, le=5000)):
    p=service(); frame=p.measurements
    for col, val in [("plate",plate),("scan_id",scan_id),("status",status),("well",well)]:
        if val: frame=frame[frame[col].astype(str).str.upper()==val.upper()]
    return {"count":len(frame), "items":records(frame.head(limit))}


@app.get("/summary")
def summary(plate: str|None=None):
    p=service(); frame=p.measurements if not plate else p.measurements[p.measurements.plate.str.upper()==plate.upper()]
    if frame.empty: raise HTTPException(404, "Plate not found")
    agg=frame.groupby("plate").agg(scans=("scan_id","nunique"), wells=("well","count"), mean_uL=("estimated_uL","mean"), mean_abs_dev_pct=("deviation_pct",lambda x:x.abs().mean())).reset_index()
    return {"items":records(agg)}


@app.post("/query")
def query(request: QueryRequest): return service().answer(request.question, request.use_llm)


@app.get("/graph")
def graph(node_type: str|None=None, limit: int=Query(1000, ge=1, le=10000)):
    p=service(); nodes=[{"id":n, **{k:json_value(v) for k,v in d.items()}} for n,d in p.graph.nodes(data=True) if not node_type or d.get("type")==node_type]
    selected={x["id"] for x in nodes[:limit]}; edges=[{"source":a,"target":b,**{k:json_value(v) for k,v in d.items()}} for a,b,d in p.graph.edges(data=True) if a in selected and b in selected]
    return {"nodes":nodes[:limit],"edges":edges}
