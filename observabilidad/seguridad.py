"""Protocolos de seguridad y uso responsable (IL3.3).

- redactar_pii: enmascara datos personales y secretos antes de guardarlos en logs o memoria.
- detectar_inyeccion: identifica intentos de prompt injection / jailbreak en la entrada.
- filtrar_salida: evita que la respuesta filtre secretos o el prompt de sistema.
"""
from __future__ import annotations

import re
import unicodedata

# (patrón, reemplazo). El orden importa: primero secretos, luego identificadores.
_PATRONES_PII = [
    (re.compile(r"gsk_[A-Za-z0-9]{16,}"), "[API_KEY]"),
    (re.compile(r"\b(sk|pk|hf)_[A-Za-z0-9]{16,}\b"), "[API_KEY]"),
    (re.compile(r"(?i)\b(contrase(?:ñ|n)a|clave|password|pass|pin)\b(\s*(?:es|era|:|=)\s*)([^\s,;.]+)"),
     lambda m: f"{m.group(1)}{m.group(2)}[SECRETO]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"), "[RUT]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[TARJETA]"),
    (re.compile(r"(?<!\d)(?:\+?56\s?)?9\s?\d{4}\s?\d{4}(?!\d)"), "[TELEFONO]"),
]


def redactar_pii(texto: str) -> str:
    """Reemplaza datos personales y secretos por etiquetas. No modifica el resto del texto."""
    if not texto:
        return texto
    for patron, reemplazo in _PATRONES_PII:
        texto = patron.sub(reemplazo, texto)
    return texto


def contiene_pii(texto: str) -> list[str]:
    """Devuelve las categorías de PII detectadas (para métricas)."""
    red = redactar_pii(texto or "")
    return sorted(set(re.findall(r"\[(API_KEY|SECRETO|EMAIL|RUT|TARJETA|TELEFONO)\]", red)))


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


_PATRONES_INYECCION = [
    r"ignora\w*\s+(todas\s+)?(las\s+|tus\s+)?(instrucciones|reglas)",
    r"olvida\w*\s+(todas\s+)?(las\s+|tus\s+)?(instrucciones|reglas)",
    r"ignore\s+(all\s+|your\s+|previous\s+)*(instructions|rules)",
    r"(prompt|instrucciones)\s+(de\s+|del\s+)?sistema",
    r"system\s+prompt",
    r"modo\s+(desarrollador|developer|dios|sin\s+restricciones)",
    r"\bjailbreak\b|\bdan\b",
    r"(revela|muestra|dime|imprime)\w*\s+.*(api\s*key|clave\s+de\s+api|token|credenciales\s+del\s+sistema)",
    r"actua\w*\s+como\s+(si\s+)?(no\s+tuvieras|un\s+administrador|root)",
]


def detectar_inyeccion(texto: str) -> list[str]:
    """Devuelve los patrones de inyección encontrados (lista vacía = entrada limpia)."""
    t = _normalizar(texto)
    return [p for p in _PATRONES_INYECCION if re.search(p, t)]


# Fragmentos que solo existen en los prompts internos: si aparecen en la salida, hay fuga.
FRAGMENTOS_PROMPT = ("REGLAS DE DECISIÓN", "Reglas de planificación", "PLAN VIGENTE", "módulo PLANIFICADOR")

MENSAJE_BLOQUEO = ("No puedo ayudarte con esa solicitud. Soy el agente de Soporte TI de Duoc UC y solo respondo "
                   "consultas técnicas basadas en la documentación oficial. ¿Tienes algún problema técnico en "
                   "el que pueda ayudarte?")


def filtrar_salida(texto: str) -> tuple[str, list[str]]:
    """Revisa la respuesta final. Devuelve (texto_seguro, motivos_de_bloqueo)."""
    motivos = []
    if any(f.lower() in (texto or "").lower() for f in FRAGMENTOS_PROMPT):
        motivos.append("fuga_prompt_sistema")
    limpio = redactar_pii(texto or "")
    if "[API_KEY]" in limpio and "[API_KEY]" not in (texto or ""):
        motivos.append("fuga_api_key")
    if "fuga_prompt_sistema" in motivos:
        return MENSAJE_BLOQUEO, motivos
    return limpio, motivos
