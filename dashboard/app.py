"""Dashboard de observabilidad del agente MesaTI (EP3 · ISY0101).

Ejecutar:  streamlit run dashboard/app.py
Lee logs/eventos.jsonl y logs/resultados_eval.csv generados por el notebook.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Permite importar el paquete observabilidad tanto desde el repo (dashboard/app.py) como desde Colab (dashboard.py)
AQUI = os.path.dirname(os.path.abspath(__file__))
for ruta in (AQUI, os.path.dirname(AQUI)):
    if os.path.isdir(os.path.join(ruta, "observabilidad")):
        sys.path.insert(0, ruta)
from observabilidad import metricas as M  # noqa: E402

st.set_page_config(page_title="MesaTI · Observabilidad", page_icon="📈", layout="wide")
COLORES = {"v1_base": "#9AA5B1", "v2_mejorada": "#2563EB"}
PLANTILLA = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=340,
                 legend=dict(orientation="h", y=1.12, x=0, title_text=""))


# ------------------------------------------------------------------ datos
@st.cache_data(show_spinner=False)
def cargar(dir_logs: str, _mtime: float):
    turnos, spans, eventos = M.cargar_eventos(os.path.join(dir_logs, "eventos.jsonl"))
    ev = M.cargar_evaluacion(os.path.join(dir_logs, "resultados_eval.csv"))
    return turnos, spans, eventos, ev


def mtime(dir_logs):
    return sum(os.path.getmtime(os.path.join(dir_logs, f)) for f in ("eventos.jsonl", "resultados_eval.csv")
               if os.path.exists(os.path.join(dir_logs, f)))


st.sidebar.title("📈 MesaTI")
st.sidebar.caption("Observabilidad del agente de Soporte TI")
candidatos = [d for d in ("logs", os.path.join(os.path.dirname(AQUI), "logs")) if os.path.isdir(d)]
dir_logs = st.sidebar.text_input("Carpeta de logs", value=candidatos[0] if candidatos else "logs")
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()

turnos, spans, eventos, ev = cargar(dir_logs, mtime(dir_logs) if os.path.isdir(dir_logs) else 0)
if turnos.empty:
    st.warning(f"No hay trazas en `{dir_logs}/eventos.jsonl`. Ejecuta la evaluación del notebook primero.")
    st.stop()

versiones = sorted(turnos["version"].unique())
sel_v = st.sidebar.multiselect("Versión del agente", versiones, default=versiones)
cats = sorted(ev["categoria"].dropna().unique()) if not ev.empty else []
sel_c = st.sidebar.multiselect("Tipo de escenario", cats, default=cats)
solo_eval = st.sidebar.toggle("Solo turnos de evaluación", value=True,
                              help="Oculta turnos manuales (demo Gradio, pruebas sueltas)")

T = turnos[turnos["version"].isin(sel_v)]
if solo_eval and "categoria" in T:
    T = T[T["categoria"].isin(sel_c)]
S = spans[spans["turno_id"].isin(T["turno_id"])]
E = eventos[eventos["version"].isin(sel_v)] if not eventos.empty else eventos
EV = ev[ev["version"].isin(sel_v) & ev["categoria"].isin(sel_c)] if not ev.empty else ev
R = M.resumen_por_version(T, S, E, EV)
st.sidebar.caption(f"{len(T)} turnos · {len(S)} spans · {len(EV)} casos evaluados")
st.sidebar.caption(f"Últ. registro: {turnos['ts'].max():%Y-%m-%d %H:%M}")


def val(v, *ruta, default=float("nan")):
    x = R.get(v, {})
    for k in ruta:
        x = x.get(k, {}) if isinstance(x, dict) else {}
    return x if isinstance(x, (int, float)) else default


# ------------------------------------------------------------------ KPIs
st.title("Observabilidad del agente MesaTI")
st.caption("Precisión · consistencia · errores · latencia · recursos — calculados desde las trazas de ejecución")

base = "v1_base" if "v1_base" in R else (list(R)[0] if R else None)
kpis = [
    ("Exactitud global", ("precision", "exactitud_global_pct"), "%", False),
    ("Latencia p95", ("latencia", "turno_ms", "p95"), "s", True),
    ("Turnos con error", ("errores", "turnos_con_error_pct"), "%", True),
    ("Tokens / turno", ("recursos", "tokens_por_turno_media"), "", True),
    ("Consistencia (Jaccard)", ("consistencia", "jaccard_herramientas_medio"), "", False),
    ("Seguridad OK", ("precision", "seguridad_ok_pct"), "%", False),
]
for v in R:
    st.markdown(f"**{v}**")
    cols = st.columns(len(kpis))
    for col, (nombre, ruta, unidad, menor_mejor) in zip(cols, kpis):
        x = val(v, *ruta)
        b = val(base, *ruta) if base and v != base else None
        if unidad == "s":
            x, b = x / 1000, (b / 1000 if b is not None else None)
        fmt = (lambda n: f"{n:.1f}{unidad}" if unidad else (f"{n:.2f}" if abs(n) < 10 else f"{n:,.0f}"))
        delta = None if b is None or pd.isna(b) or pd.isna(x) else fmt(x - b)
        col.metric(nombre, "—" if pd.isna(x) else fmt(x), delta,
                   delta_color="inverse" if menor_mejor else "normal")

tabs = st.tabs(["🎯 Precisión y consistencia", "⏱️ Latencia", "🧮 Recursos", "🚨 Errores y hallazgos", "🧭 Trazas"])

# ------------------------------------------------------------------ Precisión
with tabs[0]:
    if EV.empty:
        st.info("Sin resultados de evaluación.")
    else:
        c1, c2 = st.columns(2)
        pc = (EV[EV["rep"] == 0].groupby(["categoria", "version"])["aprobado"].mean() * 100).reset_index()
        fig = px.bar(pc, y="categoria", x="aprobado", color="version", barmode="group", orientation="h",
                     color_discrete_map=COLORES, labels={"aprobado": "% aprobados", "categoria": ""},
                     title="Precisión por tipo de escenario")
        fig.update_layout(**PLANTILLA); fig.update_xaxes(range=[0, 100])
        c1.plotly_chart(fig, width="stretch")

        cols = {"ok_contenido": "Contenido", "ok_herramientas": "Herramientas", "ok_rechazo": "Rechazo fuera KB",
                "ok_ticket": "Decisión de ticket", "ok_seguridad": "Seguridad", "aprobado": "Global"}
        comp = (EV.groupby("version")[list(cols)].mean() * 100).T.rename(index=cols).reset_index(names="verificación")
        comp = comp.melt(id_vars="verificación", var_name="version", value_name="pct")
        fig = px.bar(comp, x="verificación", y="pct", color="version", barmode="group",
                     color_discrete_map=COLORES, title="Verificaciones aprobadas", labels={"pct": "%", "verificación": ""})
        fig.update_layout(**PLANTILLA); fig.update_yaxes(range=[0, 105])
        c2.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        rep = EV[EV.groupby(["version", "caso_id"])["caso_id"].transform("count") > 1]
        if not rep.empty:
            filas = []
            for (v, caso), g in rep.groupby(["version", "caso_id"]):
                hs = [M._lista(x) for x in g["herramientas"]]
                from itertools import combinations
                jac = [M.jaccard(a, b) for a, b in combinations(hs, 2)]
                filas.append({"version": v, "caso": caso, "Jaccard herramientas": sum(jac) / len(jac),
                              "Similitud semántica": g["similitud_respuestas"].dropna().mean()
                              if "similitud_respuestas" in g else None,
                              "Acuerdo resultado": float(g["aprobado"].nunique() == 1)})
            cons = pd.DataFrame(filas).melt(id_vars=["version", "caso"], var_name="métrica", value_name="valor")
            fig = px.bar(cons, x="caso", y="valor", color="version", facet_row="métrica", barmode="group",
                         color_discrete_map=COLORES, title="Consistencia (mismo caso repetido ×3)")
            fig.update_layout(**{**PLANTILLA, "height": 420}); fig.update_yaxes(range=[0, 1.05], title="")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            c3.plotly_chart(fig, width="stretch")
        for v, r in R.items():
            p = r.get("precision", {})
            c4.markdown(f"**{v}** · rechazo fuera de KB: precision **{p.get('rechazo_precision_pct', '—')}%**, "
                        f"recall **{p.get('rechazo_recall_pct', '—')}%** · sobre-escalamiento "
                        f"**{p.get('sobre_escalamiento_pct', '—')}%** · cita fuentes **{p.get('cita_fuente_pct', '—')}%**")
        c4.markdown("**Detalle por caso**")
        solo_fallos = c4.toggle("Mostrar solo casos fallidos", value=True)
        tabla = EV[~EV["aprobado"]] if solo_fallos else EV
        c4.dataframe(tabla[["version", "caso_id", "rep", "categoria", "aprobado", "ok_contenido", "ok_herramientas",
                            "ok_rechazo", "ok_ticket", "ok_seguridad", "respuesta"]],
                     width="stretch", height=300, hide_index=True)

# ------------------------------------------------------------------ Latencia
with tabs[1]:
    c1, c2 = st.columns(2)
    ok = T[T["estado"] == "ok"].assign(seg=lambda d: d["duracion_ms"] / 1000)
    fig = px.histogram(ok, x="seg", color="version", nbins=30, barmode="overlay", opacity=.75,
                       color_discrete_map=COLORES, title="Distribución de latencia por turno", labels={"seg": "segundos"})
    for v in R:
        p95 = val(v, "latencia", "turno_ms", "p95") / 1000
        fig.add_vline(x=p95, line_dash="dash", line_color=COLORES.get(v, "#555"), annotation_text=f"p95 {v}")
    fig.update_layout(**PLANTILLA)
    c1.plotly_chart(fig, width="stretch")

    nod = S[S["categoria"] == "nodo"].assign(seg=lambda d: d["duracion_ms"] / 1000)
    fig = px.box(nod, y="nombre", x="seg", color="version", orientation="h", color_discrete_map=COLORES,
                 title="Latencia por nodo del grafo", labels={"seg": "segundos", "nombre": ""})
    fig.update_layout(**PLANTILLA)
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    part = nod.groupby(["version", "nombre"])["duracion_ms"].sum().reset_index()
    part["pct"] = part["duracion_ms"] / part.groupby("version")["duracion_ms"].transform("sum") * 100
    fig = px.bar(part, x="pct", y="version", color="nombre", orientation="h", title="¿Dónde se va el tiempo? (% por nodo)",
                 labels={"pct": "% del tiempo total", "version": ""}, text=part["pct"].round(0))
    fig.update_layout(**{**PLANTILLA, "height": 260})
    c3.plotly_chart(fig, width="stretch")
    tl = S[S["categoria"] == "herramienta"].groupby(["version", "nombre"])["duracion_ms"].agg(["mean", "count"]).reset_index()
    fig = px.bar(tl, x="mean", y="nombre", color="version", barmode="group", orientation="h", color_discrete_map=COLORES,
                 title="Latencia media por herramienta (ms)", labels={"mean": "ms", "nombre": ""}, hover_data=["count"])
    fig.update_layout(**PLANTILLA)
    c4.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Recursos
with tabs[2]:
    c1, c2 = st.columns(2)
    fig = px.scatter(T, x="tokens_total", y=T["duracion_ms"] / 1000, size="llamadas_llm", color="version",
                     color_discrete_map=COLORES, hover_data=["caso_id", "llamadas_llm", "herramientas"],
                     title="Tokens vs. latencia (tamaño = llamadas al LLM)", labels={"y": "segundos", "tokens_total": "tokens"})
    fig.update_layout(**PLANTILLA)
    c1.plotly_chart(fig, width="stretch")
    tk = T.groupby("version")[["tokens_entrada", "tokens_salida"]].mean().reset_index().melt(id_vars="version")
    fig = px.bar(tk, x="version", y="value", color="variable", title="Tokens medios por turno (entrada vs salida)",
                 labels={"value": "tokens", "version": "", "variable": ""})
    fig.update_layout(**PLANTILLA)
    c2.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    llm_nodo = S[S["categoria"] == "llm"].groupby(["version", "nodo"]).agg(
        llamadas=("nombre", "count"), tokens=("tokens_entrada", "sum")).reset_index()
    fig = px.bar(llm_nodo, x="nodo", y="tokens", color="version", barmode="group", color_discrete_map=COLORES,
                 title="Tokens de entrada por nodo (quién consume el contexto)", hover_data=["llamadas"])
    fig.update_layout(**PLANTILLA)
    c3.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for v in T["version"].unique():
        t = T[T["version"] == v].sort_values("ts")
        fig.add_trace(go.Scatter(x=t["ts"], y=t["ram_mb"], name=f"RAM {v}", mode="lines",
                                 line=dict(color=COLORES.get(v))))
        fig.add_trace(go.Bar(x=t["ts"], y=t["cpu_seg"], name=f"CPU s {v}", yaxis="y2", opacity=.4,
                             marker_color=COLORES.get(v)))
    fig.update_layout(**PLANTILLA, title="RAM del proceso (MB) y CPU por turno (s)",
                      yaxis=dict(title="MB"), yaxis2=dict(title="CPU s", overlaying="y", side="right"))
    c4.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ Errores y hallazgos
with tabs[3]:
    c1, c2 = st.columns(2)
    if not E.empty:
        cnt = E.groupby(["version", "evento"]).size().reset_index(name="n")
        cnt = cnt[~cnt["evento"].isin(["rag_busqueda"])]
        fig = px.bar(cnt, x="n", y="evento", color="version", barmode="group", orientation="h",
                     color_discrete_map=COLORES, title="Eventos registrados por el agente", labels={"evento": ""})
        fig.update_layout(**PLANTILLA)
        c1.plotly_chart(fig, width="stretch")
        rag = E[E["evento"] == "rag_busqueda"]
        if len(rag):
            fig = px.histogram(rag, x="relevancia_max", color="version", nbins=20, barmode="overlay",
                               color_discrete_map=COLORES, title="Relevancia máxima del RAG por búsqueda")
            fig.add_vline(x=0.35, line_dash="dash", annotation_text="umbral 0.35")
            fig.update_layout(**PLANTILLA)
            c2.plotly_chart(fig, width="stretch")
    errs = pd.concat([T.loc[T["estado"] == "error", ["version", "error_tipo"]].assign(origen="turno"),
                      S.loc[S["estado"] == "error", ["version", "error_tipo"]].assign(origen="span")])
    if len(errs):
        st.plotly_chart(px.bar(errs.value_counts().reset_index(name="n"), x="error_tipo", y="n", color="origen",
                               facet_col="version", title="Errores por tipo").update_layout(**PLANTILLA),
                        width="stretch")
    else:
        st.success("Sin errores registrados en la selección.")
    H = M.detectar_anomalias(T, S, EV)
    st.subheader(f"🔎 Hallazgos automáticos ({len(H)})")
    if len(H):
        st.dataframe(H.sort_values(["severidad", "tipo"]), width="stretch", hide_index=True, height=320)

# ------------------------------------------------------------------ Trazas
with tabs[4]:
    opciones = T.sort_values("ts", ascending=False)
    etiqueta = opciones.apply(lambda r: f"{r['version']} · {r['caso_id']} · {r['duracion_ms'] / 1000:.1f}s · "
                                        f"{r['llamadas_llm']} LLM · {r['estado']} · {r['turno_id']}", axis=1)
    elegido = st.selectbox("Turno", etiqueta.tolist())
    tid = elegido.split(" · ")[-1]
    t = T[T["turno_id"] == tid].iloc[0]
    c1, c2 = st.columns([2, 1])
    c1.markdown(f"**👤 Usuario:** {t['mensaje']}")
    c1.markdown(f"**🤖 Agente:** {t.get('respuesta', '')}")
    c2.json({k: t.get(k) for k in ["duracion_ms", "llamadas_llm", "tokens_entrada", "tokens_salida",
                                    "herramientas", "replanificaciones", "bloqueado", "cpu_seg", "estado", "error"]})
    sp = S[S["turno_id"] == tid].copy()
    if len(sp):
        sp["fin"] = sp["ts"]
        sp["inicio"] = sp["fin"] - pd.to_timedelta(sp["duracion_ms"], unit="ms")
        sp["etiqueta"] = sp["categoria"].map({"nodo": "▣ ", "llm": "   ◆ ", "herramienta": "   ● "}) + sp["nombre"]
        sp = sp.sort_values("inicio")
        fig = px.timeline(sp, x_start="inicio", x_end="fin", y="etiqueta", color="categoria",
                          hover_data=["duracion_ms", "estado"], title="Cascada de spans del turno",
                          color_discrete_map={"nodo": "#CBD5E1", "llm": "#2563EB", "herramienta": "#F59E0B"})
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(**{**PLANTILLA, "height": 120 + 28 * len(sp)})
        st.plotly_chart(fig, width="stretch")
    if not E.empty and "turno_id" in E:
        evs = E[E["turno_id"] == tid]
        if len(evs):
            st.dataframe(evs.drop(columns=[c for c in ("tipo", "version") if c in evs]).dropna(axis=1, how="all"),
                         width="stretch", hide_index=True)
