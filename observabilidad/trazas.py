"""Trazabilidad del agente MesaTI.

Registra cada ejecución del agente como un árbol de *spans* (turno → nodo → LLM/herramienta),
con latencia, tokens, errores y uso de recursos. Los eventos se guardan en formato JSONL
(una línea por evento), compatible con herramientas como Grafana Loki, Kibana o pandas.

Uso:
    obs = Observador("logs/eventos.jsonl", version="v1_base")
    with obs.turno(caso_id="C01", user_id="u1", thread_id="t1", mensaje="hola") as t:
        resultado = agente.invoke(entrada, {"callbacks": [t.callback], ...})
        t.cerrar(respuesta=..., estado=resultado)
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

try:
    import psutil
except ImportError:  # psutil es opcional; sin él no se mide CPU/RAM
    psutil = None

from .seguridad import redactar_pii


def _ahora_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _clasificar_error(exc: BaseException | str) -> str:
    """Agrupa errores en categorías útiles para el dashboard."""
    texto = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
    t = texto.lower()
    if "429" in t or "rate limit" in t or "ratelimit" in t or "rate_limit" in t:
        return "rate_limit"
    if "recursion" in t:
        return "limite_recursion"
    if "timeout" in t or "timed out" in t:
        return "timeout"
    if "401" in t or "invalid api key" in t or "authentication" in t:
        return "autenticacion"
    if "validation" in t or "pydantic" in t or "parse" in t or "json" in t:
        return "salida_mal_formada"
    if "connection" in t or "network" in t:
        return "red"
    return "otro"


class Observador:
    """Escribe eventos de observabilidad en un archivo JSONL (thread-safe)."""

    def __init__(self, ruta: str = "logs/eventos.jsonl", version: str = "v1_base", redactar: bool = True):
        self.ruta = ruta
        self.version = version
        self.redactar = redactar
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.turno_actual: Optional[TurnoTraza] = None

    # ---------------- escritura ----------------
    def _limpiar(self, valor: Any) -> Any:
        if not self.redactar:
            return valor
        if isinstance(valor, str):
            return redactar_pii(valor)
        if isinstance(valor, dict):
            return {k: self._limpiar(v) for k, v in valor.items()}
        if isinstance(valor, list):
            return [self._limpiar(v) for v in valor]
        return valor

    def escribir(self, evento: dict) -> None:
        evento = {"ts": _ahora_iso(), "version": self.version, **evento}
        evento = self._limpiar(evento)
        linea = json.dumps(evento, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.ruta, "a", encoding="utf-8") as f:
                f.write(linea + "\n")

    def evento(self, tipo: str, **datos) -> None:
        """Evento puntual (p. ej. replanificación, guardrail, plan por defecto)."""
        base = {"tipo": "evento", "evento": tipo}
        if self.turno_actual is not None:
            base.update(turno_id=self.turno_actual.turno_id, caso_id=self.turno_actual.caso_id)
        self.escribir({**base, **datos})

    # ---------------- turnos ----------------
    @contextmanager
    def turno(self, caso_id: str = "", user_id: str = "", thread_id: str = "", mensaje: str = "", **extra):
        t = TurnoTraza(self, caso_id=caso_id, user_id=user_id, thread_id=thread_id, mensaje=mensaje, extra=extra)
        self.turno_actual = t
        try:
            yield t
        except Exception as e:  # el error del turno queda registrado y se relanza
            t.error = e
            raise
        finally:
            t._finalizar()
            self.turno_actual = None


class TurnoTraza:
    """Una interacción completa usuario → agente."""

    def __init__(self, obs: Observador, caso_id, user_id, thread_id, mensaje, extra):
        self.obs = obs
        self.turno_id = uuid.uuid4().hex[:12]
        self.caso_id = caso_id
        self.user_id = user_id
        self.thread_id = thread_id
        self.mensaje = mensaje
        self.extra = extra
        self.error: Optional[BaseException] = None
        self.respuesta = ""
        self.datos_cierre: dict = {}
        self.callback = TrazaCallback(obs, self)
        self._t0 = time.perf_counter()
        self._proc = psutil.Process() if psutil else None
        self._cpu0 = self._cpu_seg()
        self._rss0 = self._rss_mb()

    def _cpu_seg(self) -> float:
        if not self._proc:
            return 0.0
        c = self._proc.cpu_times()
        return c.user + c.system

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / 1_048_576 if self._proc else 0.0

    def cerrar(self, respuesta: str = "", **datos) -> None:
        """Llamar al terminar con éxito: guarda la respuesta y datos extra (herramientas usadas, etc.)."""
        self.respuesta = respuesta
        self.datos_cierre.update(datos)

    def _finalizar(self) -> None:
        dur_ms = (time.perf_counter() - self._t0) * 1000
        cb = self.callback
        evento = {
            "tipo": "turno", "turno_id": self.turno_id, "caso_id": self.caso_id,
            "user_id": self.user_id, "thread_id": self.thread_id,
            "mensaje": self.mensaje, "respuesta": self.respuesta[:2000],
            "duracion_ms": round(dur_ms, 1),
            "estado": "error" if self.error else "ok",
            "error_tipo": _clasificar_error(self.error) if self.error else None,
            "error": f"{type(self.error).__name__}: {self.error}"[:500] if self.error else None,
            "llamadas_llm": cb.llamadas_llm,
            "llamadas_herramientas": cb.llamadas_herramientas,
            "herramientas": cb.herramientas_usadas,
            "errores_herramientas": cb.errores_herramientas,
            "tokens_entrada": cb.tokens_entrada,
            "tokens_salida": cb.tokens_salida,
            "tokens_total": cb.tokens_entrada + cb.tokens_salida,
            "cpu_seg": round(self._cpu_seg() - self._cpu0, 3),
            "ram_mb": round(self._rss_mb(), 1),
            "ram_delta_mb": round(self._rss_mb() - self._rss0, 1),
            **self.extra, **self.datos_cierre,
        }
        self.obs.escribir(evento)


class TrazaCallback(BaseCallbackHandler):
    """Callback de LangChain/LangGraph que convierte la ejecución en spans."""

    raise_error = False  # la observabilidad nunca debe romper al agente

    def __init__(self, obs: Observador, turno: TurnoTraza):
        self.obs = obs
        self.turno = turno
        self._inicio: dict[UUID, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self.llamadas_llm = 0
        self.llamadas_herramientas = 0
        self.errores_herramientas = 0
        self.herramientas_usadas: list[str] = []
        self.senales: list[str] = []          # [SIN_RESULTADOS] / [REPLANIFICAR] emitidas por herramientas
        self.tokens_entrada = 0
        self.tokens_salida = 0

    # ---------- utilidades ----------
    def _abrir(self, run_id: UUID, info: dict) -> None:
        with self._lock:
            self._inicio[run_id] = (time.perf_counter(), info)

    def _cerrar(self, run_id: UUID, estado: str = "ok", **datos) -> None:
        with self._lock:
            par = self._inicio.pop(run_id, None)
        if par is None:
            return
        t0, info = par
        self.obs.escribir({
            "tipo": "span", "turno_id": self.turno.turno_id, "caso_id": self.turno.caso_id,
            "span_id": str(run_id)[:8], **info,
            "duracion_ms": round((time.perf_counter() - t0) * 1000, 1), "estado": estado, **datos,
        })

    @staticmethod
    def _nodo(metadata: Optional[dict]) -> str:
        return (metadata or {}).get("langgraph_node", "") or ""

    # ---------- nodos del grafo ----------
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kw):
        nombre = kw.get("name") or (serialized or {}).get("name", "")
        nodo = self._nodo(metadata)
        # Solo registramos el span del nodo en sí (no las cadenas internas)
        if nodo and nombre == nodo:
            self._abrir(run_id, {"categoria": "nodo", "nombre": nodo, "nodo": nodo,
                                 "paso": (metadata or {}).get("langgraph_step")})

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._cerrar(run_id)

    def on_chain_error(self, error, *, run_id, **kw):
        self._cerrar(run_id, estado="error", error_tipo=_clasificar_error(error), error=str(error)[:300])

    # ---------- llamadas al LLM ----------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kw):
        n_chars = sum(len(str(getattr(m, "content", ""))) for lote in messages for m in lote)
        self._abrir(run_id, {"categoria": "llm", "nombre": f"llm@{self._nodo(metadata) or '?'}",
                             "nodo": self._nodo(metadata), "chars_prompt": n_chars,
                             "modelo": (metadata or {}).get("ls_model_name")})

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kw):
        self._abrir(run_id, {"categoria": "llm", "nombre": f"llm@{self._nodo(metadata) or '?'}",
                             "nodo": self._nodo(metadata), "chars_prompt": sum(len(p) for p in prompts)})

    def on_llm_end(self, response, *, run_id, **kw):
        t_in = t_out = 0
        n_tool_calls = 0
        try:
            msg = response.generations[0][0].message
            uso = getattr(msg, "usage_metadata", None) or {}
            t_in, t_out = int(uso.get("input_tokens", 0)), int(uso.get("output_tokens", 0))
            n_tool_calls = len(getattr(msg, "tool_calls", []) or [])
        except Exception:
            pass
        if not (t_in or t_out):  # respaldo: algunos proveedores lo ponen en llm_output
            uso = (response.llm_output or {}).get("token_usage", {}) if response.llm_output else {}
            t_in, t_out = int(uso.get("prompt_tokens", 0)), int(uso.get("completion_tokens", 0))
        with self._lock:
            self.llamadas_llm += 1
            self.tokens_entrada += t_in
            self.tokens_salida += t_out
        self._cerrar(run_id, tokens_entrada=t_in, tokens_salida=t_out, tool_calls=n_tool_calls)

    def on_llm_error(self, error, *, run_id, **kw):
        with self._lock:
            self.llamadas_llm += 1
        self._cerrar(run_id, estado="error", error_tipo=_clasificar_error(error), error=str(error)[:300])

    # ---------- herramientas ----------
    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kw):
        nombre = kw.get("name") or (serialized or {}).get("name", "herramienta")
        with self._lock:
            self.llamadas_herramientas += 1
            self.herramientas_usadas.append(nombre)
        self._abrir(run_id, {"categoria": "herramienta", "nombre": nombre, "nodo": self._nodo(metadata),
                             "entrada": str(input_str)[:300]})

    def on_tool_end(self, output, *, run_id, **kw):
        texto = str(getattr(output, "content", output))
        señales = [s for s in ("[SIN_RESULTADOS]", "[REPLANIFICAR]") if s in texto]
        with self._lock:
            self.senales.extend(señales)
        self._cerrar(run_id, salida=texto[:300], senales=señales)

    def on_tool_error(self, error, *, run_id, **kw):
        with self._lock:
            self.errores_herramientas += 1
        self._cerrar(run_id, estado="error", error_tipo=_clasificar_error(error), error=str(error)[:300])
