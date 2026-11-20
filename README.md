# MesaTI · Observabilidad de un agente de Soporte TI

**ISY0101 – Ingeniería de Soluciones con IA · Evaluación Parcial N°3 · Duoc UC**

MesaTI es un agente de Soporte TI construido con **LangGraph + RAG + herramientas + memoria** (EP1 y EP2).
En esta entrega se le agrega una **capa de observabilidad**: cada ejecución queda registrada como una traza
(turno → nodo → llamada al LLM / herramienta) y, a partir de esas trazas, se miden **precisión, consistencia,
frecuencia de errores, latencia y uso de recursos**. Con esos datos se construyó una versión mejorada (v2) y se
comparó contra la original (v1).

```
v1_base:      START → recuperar_memoria → planificador → agente ⇄ herramientas → memorizar → END
v2_mejorada:  START → guardrail_entrada ─(bloqueo)→ END
                         └→ recuperar_memoria → planificador → agente ⇄ herramientas → guardrail_salida → memorizar → END

                  TrazaCallback (LangChain callbacks)
                           │  spans: nodo · llm · herramienta  (+ tokens, errores, CPU/RAM)
                           ▼
                  logs/eventos.jsonl  ──►  observabilidad/metricas.py  ──►  dashboard (Streamlit)
                  logs/resultados_eval.csv                              └─►  reportes/ (tablas y gráficos)
```

## Estructura

| Ruta | Contenido |
|---|---|
| `notebooks/EP3_Observabilidad_MesaTI.ipynb` | Notebook principal (Colab): agente, evaluación v1/v2, métricas, análisis de trazas, gráficos, dashboard y demo |
| `observabilidad/trazas.py` | `Observador` y `TrazaCallback`: registro de spans con latencia, tokens, errores y recursos |
| `observabilidad/seguridad.py` | Enmascarado de datos personales/secretos, detección de *prompt injection*, filtro de salida |
| `observabilidad/metricas.py` | Cálculo de métricas y detección automática de anomalías |
| `dashboard/app.py` | Dashboard interactivo (Streamlit + Plotly) |
| `tests/test_observabilidad.py` | Pruebas automáticas (pytest, sin API key) |
| `logs/` | Trazas y resultados de la evaluación (se generan al ejecutar el notebook) |
| `reportes/` | Tablas (`comparativa_v1_v2.csv`, `hallazgos.csv`, `resumen_metricas.json`) y gráficos PNG |
| `notebooks/anteriores/` | Notebooks de la EP1 y EP2 (claves eliminadas) |

## Cómo ejecutar

### 1. Notebook (Google Colab, recomendado)
1. Abre `notebooks/EP3_Observabilidad_MesaTI.ipynb` en Colab.
2. En el panel 🔑 *Secrets* crea `GROQ_API_KEY` con tu clave de [console.groq.com](https://console.groq.com) y habilita el acceso del notebook.
3. *Entorno de ejecución → Ejecutar todo*. La sección 10 evalúa v1 y v2 (≈ 30–40 min por el límite de tasa de Groq).
   Si aparece un error 429, vuelve a ejecutar la celda: la evaluación se reanuda donde quedó.
4. La sección 12 abre el dashboard con un enlace público temporal (`*.trycloudflare.com`).
5. La sección 13 descarga `evidencia_ep3.zip` con `logs/` y `reportes/`.

### 2. Local
```bash
git clone <URL-del-repo> && cd mesati-observabilidad
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # y escribe tu GROQ_API_KEY
jupyter notebook notebooks/EP3_Observabilidad_MesaTI.ipynb
```

### 3. Dashboard
Con los archivos `logs/eventos.jsonl` y `logs/resultados_eval.csv` en la carpeta `logs/`:
```bash
streamlit run dashboard/app.py
```

### 4. Pruebas
```bash
pytest -q
```

## Métricas implementadas

| Dimensión | Métrica | Cómo se mide |
|---|---|---|
| Precisión | Exactitud global, contenido correcto, herramientas correctas | 22 casos con respuesta esperada (palabras clave, herramientas obligatorias) |
| Precisión | Precisión/recall del rechazo fuera de la KB | Matriz de confusión de la frase *"no poseo esa información"* |
| Precisión | Sobre-escalamiento | % de tickets creados en consultas informativas |
| Consistencia | Jaccard de herramientas, similitud semántica, acuerdo de resultado | 5 casos repetidos ×3; misma intención con redacción distinta (ruido, inglés, coloquial) |
| Errores | % turnos con error por tipo (429, salida mal formada, timeout…), fallas de herramientas, plan por defecto | Excepciones clasificadas en los spans y eventos |
| Latencia | p50/p95/p99 por turno, por nodo, por herramienta y % del tiempo dentro del LLM | Spans del callback |
| Recursos | Tokens entrada/salida, llamadas al LLM por turno, CPU y RAM del proceso | `usage_metadata` del LLM + `psutil` |
| Seguridad | % de casos sin fuga de prompt, API key o datos personales (respuesta, memoria, perfil, tickets) | Casos adversariales C19–C22 |

## Seguridad y uso responsable
- La API key se lee de variables de entorno / Secrets; `.env` está en `.gitignore`. La clave publicada en la EP1/EP2 fue **revocada**.
- Los logs y la memoria de largo plazo se guardan **anonimizados** (RUT, email, teléfono, contraseñas, tarjetas, API keys).
- v2 bloquea intentos de *prompt injection*, no envía contraseñas al LLM externo y filtra la salida.
- La demo de Gradio exige usuario y clave.

## Uso de IA
Ver la declaración en el informe técnico.
