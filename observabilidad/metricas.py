"""Cálculo de métricas de observabilidad a partir de los logs JSONL y de la evaluación.

Lo usan tanto el notebook (para el informe) como el dashboard de Streamlit.
"""
from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- carga
def cargar_eventos(ruta: str = "logs/eventos.jsonl"):
    """Devuelve (turnos, spans, eventos) como DataFrames."""
    filas = []
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    try:
                        filas.append(json.loads(linea))
                    except json.JSONDecodeError:
                        pass
    df = pd.DataFrame(filas)
    if df.empty:
        vacio = pd.DataFrame()
        return vacio, vacio, vacio
    df["ts"] = pd.to_datetime(df["ts"])
    turnos = df[df["tipo"] == "turno"].dropna(axis=1, how="all").reset_index(drop=True)
    spans = df[df["tipo"] == "span"].dropna(axis=1, how="all").reset_index(drop=True)
    eventos = df[df["tipo"] == "evento"].dropna(axis=1, how="all").reset_index(drop=True)
    return turnos, spans, eventos


def cargar_evaluacion(ruta: str = "logs/resultados_eval.csv") -> pd.DataFrame:
    return pd.read_csv(ruta) if os.path.exists(ruta) else pd.DataFrame()


# ----------------------------------------------------------------- utilidades
def _pct(serie) -> float:
    serie = pd.Series(serie).dropna()
    return round(100 * serie.mean(), 1) if len(serie) else float("nan")


def percentiles(serie, ps=(50, 90, 95, 99)) -> dict:
    s = pd.Series(serie).dropna()
    if s.empty:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": round(float(np.percentile(s, p)), 1) for p in ps}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _lista(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try:
            return json.loads(x.replace("'", '"'))
        except Exception:
            return []
    return []


# ----------------------------------------------------------------- métricas
def metricas_precision(ev: pd.DataFrame) -> dict:
    """Precisión funcional: % de casos que cumplen cada verificación."""
    if ev.empty:
        return {}
    r = {"casos": int(len(ev)), "exactitud_global_pct": _pct(ev["aprobado"])}
    for col, nombre in [("ok_contenido", "contenido_correcto_pct"), ("ok_herramientas", "herramientas_correctas_pct"),
                        ("ok_ticket", "decision_ticket_correcta_pct"), ("ok_rechazo", "rechazo_correcto_pct"),
                        ("ok_seguridad", "seguridad_ok_pct"), ("cita_fuente", "cita_fuente_pct")]:
        if col in ev:
            r[nombre] = _pct(ev[col])
    # Matriz de confusión del rechazo ("no poseo esa información")
    if {"espera_rechazo", "rechazo"}.issubset(ev.columns):
        e = ev.dropna(subset=["espera_rechazo"])
        e = e[e["espera_rechazo"].astype(str).isin(["True", "False", "1", "0", "1.0", "0.0"])]
        esp = e["espera_rechazo"].astype(str).isin(["True", "1", "1.0"])
        obt = e["rechazo"].astype(bool)
        vp, fp, fn = int((esp & obt).sum()), int((~esp & obt).sum()), int((esp & ~obt).sum())
        r["rechazo_precision_pct"] = round(100 * vp / (vp + fp), 1) if vp + fp else float("nan")
        r["rechazo_recall_pct"] = round(100 * vp / (vp + fn), 1) if vp + fn else float("nan")
    if {"espera_ticket", "creo_ticket"}.issubset(ev.columns):
        e = ev[ev["espera_ticket"].astype(str).isin(["False", "0", "0.0"])]
        r["sobre_escalamiento_pct"] = _pct(e["creo_ticket"].astype(bool)) if len(e) else float("nan")
    return r


def metricas_consistencia(ev: pd.DataFrame) -> dict:
    """Consistencia: mismo caso repetido y mismo intento con distinta redacción."""
    if ev.empty:
        return {}
    r = {}
    rep = ev[ev.groupby("caso_id")["caso_id"].transform("count") > 1]
    if not rep.empty:
        jac, acuerdo, ticket_ac, sim = [], [], [], []
        for _, g in rep.groupby("caso_id"):
            hs = [_lista(x) for x in g["herramientas"]]
            jac += [jaccard(a, b) for a, b in combinations(hs, 2)]
            acuerdo.append(g["aprobado"].nunique() == 1)
            ticket_ac.append(g["creo_ticket"].nunique() == 1)
            if "similitud_respuestas" in g and g["similitud_respuestas"].notna().any():
                sim.append(g["similitud_respuestas"].dropna().iloc[0])
        r["casos_repetidos"] = int(rep["caso_id"].nunique())
        r["jaccard_herramientas_medio"] = round(float(np.mean(jac)), 3) if jac else float("nan")
        r["acuerdo_resultado_pct"] = _pct(acuerdo)
        r["acuerdo_ticket_pct"] = _pct(ticket_ac)
        r["similitud_semantica_media"] = round(float(np.mean(sim)), 3) if sim else float("nan")
    if "grupo" in ev:
        grupos = ev[ev["rep"] == 0].dropna(subset=["grupo"]).groupby("grupo")
        acuerdo_g = [g["aprobado"].nunique() == 1 for _, g in grupos if len(g) > 1]
        jac_g = []
        for _, g in grupos:
            hs = [set(_lista(x)) for x in g["herramientas"]]
            jac_g += [jaccard(a, b) for a, b in combinations(hs, 2)]
        r["acuerdo_parafrasis_pct"] = _pct(acuerdo_g)
        r["jaccard_parafrasis_medio"] = round(float(np.mean(jac_g)), 3) if jac_g else float("nan")
    return r


def metricas_errores(turnos: pd.DataFrame, spans: pd.DataFrame, eventos: pd.DataFrame) -> dict:
    if turnos.empty:
        return {}
    n = len(turnos)
    r = {"turnos": n,
         "turnos_con_error_pct": _pct(turnos["estado"] == "error"),
         "errores_por_tipo": turnos.loc[turnos["estado"] == "error", "error_tipo"].value_counts().to_dict()}
    if not spans.empty:
        llm = spans[spans["categoria"] == "llm"]
        tools = spans[spans["categoria"] == "herramienta"]
        r["llm_error_pct"] = _pct(llm["estado"] == "error") if len(llm) else 0.0
        r["herramienta_error_pct"] = _pct(tools["estado"] == "error") if len(tools) else 0.0
        if len(llm) and "error_tipo" in llm:
            r["errores_llm_por_tipo"] = llm["error_tipo"].dropna().value_counts().to_dict()
    if not eventos.empty:
        conteo = eventos["evento"].value_counts().to_dict()
        r["eventos"] = conteo
        r["plan_por_defecto_pct"] = round(100 * conteo.get("plan_por_defecto", 0) / n, 1)
        r["replanificaciones_por_turno"] = round(conteo.get("replanificacion", 0) / n, 3)
    return r


def metricas_latencia(turnos: pd.DataFrame, spans: pd.DataFrame) -> dict:
    if turnos.empty:
        return {}
    ok = turnos[turnos["estado"] == "ok"]
    r = {"turno_ms": {**percentiles(ok["duracion_ms"]), "media": round(float(ok["duracion_ms"].mean()), 1),
                      "max": round(float(ok["duracion_ms"].max()), 1)}}
    if not spans.empty:
        nodos = spans[spans["categoria"] == "nodo"]
        r["por_nodo_ms"] = nodos.groupby("nombre")["duracion_ms"].agg(
            media="mean", p95=lambda s: np.percentile(s, 95), total="sum", n="count").round(1).to_dict("index")
        total = nodos["duracion_ms"].sum()
        r["participacion_nodo_pct"] = (nodos.groupby("nombre")["duracion_ms"].sum() / total * 100).round(1).to_dict()
        llm = spans[spans["categoria"] == "llm"]
        tools = spans[spans["categoria"] == "herramienta"]
        if len(llm):
            r["llm_ms"] = {**percentiles(llm["duracion_ms"]), "media": round(float(llm["duracion_ms"].mean()), 1)}
            r["tiempo_en_llm_pct"] = round(float(100 * llm["duracion_ms"].sum() / ok["duracion_ms"].sum()), 1)
        if len(tools):
            r["por_herramienta_ms"] = tools.groupby("nombre")["duracion_ms"].mean().round(1).to_dict()
    return r


def metricas_recursos(turnos: pd.DataFrame) -> dict:
    if turnos.empty:
        return {}
    return {
        "tokens_por_turno_media": round(float(turnos["tokens_total"].mean()), 1),
        "tokens_entrada_media": round(float(turnos["tokens_entrada"].mean()), 1),
        "tokens_salida_media": round(float(turnos["tokens_salida"].mean()), 1),
        "tokens_totales": int(turnos["tokens_total"].sum()),
        "llamadas_llm_por_turno": round(float(turnos["llamadas_llm"].mean()), 2),
        "llamadas_llm_max": int(turnos["llamadas_llm"].max()),
        "herramientas_por_turno": round(float(turnos["llamadas_herramientas"].mean()), 2),
        "cpu_seg_por_turno": round(float(turnos["cpu_seg"].mean()), 3),
        "ram_mb_max": round(float(turnos["ram_mb"].max()), 1),
        "ratio_entrada_salida": round(float(turnos["tokens_entrada"].sum() / max(1, turnos["tokens_salida"].sum())), 1),
    }


def resumen_por_version(turnos, spans, eventos, ev) -> dict:
    """Todas las métricas, separadas por versión del agente (v1_base / v2_mejorada)."""
    versiones = sorted(set(turnos.get("version", pd.Series(dtype=str)).dropna()) |
                       set(ev.get("version", pd.Series(dtype=str)).dropna()))
    salida = {}
    for v in versiones:
        t = turnos[turnos["version"] == v] if not turnos.empty else turnos
        s = spans[spans["version"] == v] if not spans.empty else spans
        e = eventos[eventos["version"] == v] if not eventos.empty else eventos
        x = ev[ev["version"] == v] if not ev.empty else ev
        salida[v] = {"precision": metricas_precision(x), "consistencia": metricas_consistencia(x),
                     "errores": metricas_errores(t, s, e), "latencia": metricas_latencia(t, s),
                     "recursos": metricas_recursos(t)}
    return salida


# ----------------------------------------------------------------- análisis de trazas
def detectar_anomalias(turnos: pd.DataFrame, spans: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Reglas automáticas sobre los registros para encontrar puntos de falla o cuellos de botella."""
    hallazgos = []

    def h(severidad, tipo, turno, detalle):
        hallazgos.append({"severidad": severidad, "tipo": tipo, "version": turno.get("version"),
                          "caso_id": turno.get("caso_id"), "turno_id": turno.get("turno_id"), "detalle": detalle})

    if turnos.empty:
        return pd.DataFrame(hallazgos)
    for v, tv in turnos.groupby("version"):
        ok = tv[tv["estado"] == "ok"]
        if len(ok) >= 5:
            umbral = np.percentile(ok["duracion_ms"], 95)
            for _, t in ok[ok["duracion_ms"] > umbral].iterrows():
                dom = ""
                if not spans.empty:
                    sp = spans[(spans["turno_id"] == t["turno_id"]) & (spans["categoria"] == "nodo")]
                    if len(sp):
                        top = sp.groupby("nombre")["duracion_ms"].sum().sort_values(ascending=False)
                        dom = f"; nodo dominante: {top.index[0]} ({top.iloc[0] / t['duracion_ms'] * 100:.0f}%)"
                h("media", "latencia_atipica", t, f"{t['duracion_ms'] / 1000:.1f}s > p95 ({umbral / 1000:.1f}s){dom}")
        for _, t in tv[tv["llamadas_llm"] >= 5].iterrows():
            h("media", "exceso_llamadas_llm", t, f"{t['llamadas_llm']} llamadas al LLM en un turno (posible bucle ReAct)")
        for _, t in tv[tv["estado"] == "error"].iterrows():
            h("alta", f"error_{t.get('error_tipo')}", t, str(t.get("error"))[:160])
    if not ev.empty:
        for _, c in ev.iterrows():
            base = {"version": c.get("version"), "caso_id": c.get("caso_id"), "turno_id": c.get("turno_id")}
            if c.get("sin_resultados_kb") and not c.get("rechazo") and not c.get("creo_ticket"):
                hallazgos.append({**base, "severidad": "alta", "tipo": "posible_alucinacion",
                                  "detalle": "La KB devolvió [SIN_RESULTADOS] y el agente respondió sin la frase de rechazo ni escalar"})
            if str(c.get("espera_ticket")) in ("False", "0", "0.0") and c.get("creo_ticket"):
                hallazgos.append({**base, "severidad": "media", "tipo": "sobre_escalamiento",
                                  "detalle": "Se creó un ticket para una consulta informativa"})
            if str(c.get("espera_ticket")) in ("True", "1", "1.0") and not c.get("creo_ticket"):
                hallazgos.append({**base, "severidad": "alta", "tipo": "incidente_no_escalado",
                                  "detalle": "Incidente que requería ticket y no se registró"})
            if c.get("ok_seguridad") is False or str(c.get("ok_seguridad")) == "False":
                hallazgos.append({**base, "severidad": "alta", "tipo": "falla_seguridad",
                                  "detalle": str(c.get("motivo_seguridad", ""))[:160]})
    return pd.DataFrame(hallazgos)
