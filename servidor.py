# servidor.py
import os
import json
import re
from typing import List, Optional
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# -------------------------
# Configuración general
# -------------------------
TOP_K = 6
EMB_MODEL = "intfloat/multilingual-e5-base"

CORS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "curso" / "index" / "faiss.index"
META_PATH = BASE_DIR / "curso" / "index" / "meta.json"

# Mensaje de sistema (neutro en español por compatibilidad con /ask)
SYSTEM_MSG = (
    "Actúas como asistente del curso. Responde SOLO con el 'Contexto'. "
    "Si la pregunta excede el material, di: 'Aún no lo vimos en clase'. "
    "Usa el mismo tono que las diapositivas."
)

app = FastAPI(title="PPT-RAG-Groq")

FRONT = os.getenv("FRONT_ORIGIN", "https://runner-py-ia.vercel.app")
CORS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
origins = CORS or [FRONT]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,             # True solo si usas cookies/Auth
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# Modelos de entrada
# -------------------------
class AskBody(BaseModel):
    pregunta: str
    clase: Optional[str] = None
    # Si querés que /ask también soporte idioma, podés descomentar la línea siguiente:
    # idioma: Optional[str] = "es"

class ConsejoBody(BaseModel):
    enunciado: str
    codigo: str
    idioma: Optional[str] = "es"
    clase: Optional[str] = None  # p.ej., "PYTH_1200Funciones"

# -------------------------
# Lazy load de recursos
# -------------------------
emb_model = None
faiss_index = None
META = None
groq_client = None

def ensure_loaded():
    """Carga perezosa de emb_model, índice FAISS, metadatos y cliente Groq."""
    global emb_model, faiss_index, META, groq_client

    if emb_model is None:
        emb_model = SentenceTransformer(EMB_MODEL)

    if faiss_index is None or META is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise HTTPException(status_code=500, detail="Índice no encontrado. Corré extractor_pptx.py e ingesta_embeddings.py.")
        faiss_index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            META = json.load(f)

    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Falta GROQ_API_KEY en .env")
        groq_client = Groq(api_key=api_key)

# -------------------------
# Utilidades (búsqueda y helpers)
# -------------------------
def buscar_contexto(query: str, clase: Optional[str]) -> List[dict]:
    """Búsqueda general (opcionalmente filtrada por clase)."""
    qv = emb_model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(qv, dtype="float32"), TOP_K * 2)
    out = []
    for idx in I[0]:
        row = META[idx]
        if clase and row.get("clase") != clase:
            continue
        out.append(row)
        if len(out) >= TOP_K:
            break
    return out

def buscar_contexto_por_clase(query: str, clase: Optional[str]) -> List[dict]:
    qv = emb_model.encode([query], normalize_embeddings=True)
    D, I = faiss_index.search(np.array(qv, dtype="float32"), TOP_K * 3)
    out = []
    for idx in I[0]:
        row = META[idx]
        if clase:
            # Coincidencia flexible (si el nombre enviado está incluido en la clase de META)
            if clase not in row.get("clase", ""):
                continue
        out.append(row)
        if len(out) >= TOP_K:
            break
    return out

def prettify_clase(raw: str) -> str:
    """
    Convierte 'PYTH_1000 - C01 - Introducción_ ¿Qué es Python_ 01'
    en 'Introducción ¿Qué es Python?'.
    """
    if not raw:
        return ""
    # Quitar código inicial tipo "PYTH_1000 - C01 - "
    s = re.sub(r"^PYTH_\d+\s*-\s*C\d+\s*-\s*", "", raw)
    # Reemplazar guiones bajos por espacios
    s = s.replace("_", " ")
    # Quitar numeración final tipo " 01"
    s = re.sub(r"\s+\d+$", "", s)
    # Normalizar espacios
    s = re.sub(r"\s+", " ", s).strip()
    # Si no tiene signo de interrogación final, agregar si corresponde
    if s.endswith("Python"):
        s = s + "?"
    return s

def format_fuentes(ctx: list[dict]) -> list[str]:
    out = []
    for c in ctx:
        clase_pretty = prettify_clase(c.get("clase", ""))
        slide = c.get("slide")
        out.append(f"👉 Revisa la clase \"{clase_pretty}\" (slide {slide}).")
    return out

def construir_prompt(contextos: List[dict], pregunta: str) -> str:
    ctx = "\n\n".join([f"(Clase: {c['clase']} • Slide: {c['slide']}) {c['text']}" for c in contextos])
    return (
        "Contexto:\n"
        f"{ctx}\n\n"
        "Pregunta del alumno:\n"
        f"{pregunta}\n\n"
        "Instrucciones:\n"
        "- Usa solo el Contexto.\n"
        "- Incluye al final una sección \"Fuentes\" listando (Clase y Slide) usados.\n"
    )

def codeblocks_to_html(text: str) -> str:
    """Convierte ```python ... ``` en <pre><code class='language-python'>...</code></pre>."""
    return re.sub(
        r"```python(.*?)```",
        r"<pre><code class='language-python'>\1</code></pre>",
        text,
        flags=re.DOTALL,
    )

# ---- Guardrails extra ----
def is_blank(s: Optional[str]) -> bool:
    return not s or not s.strip()

ADVANCED_TERMS = [
    "lista", "listas", "list comprehension", "comprehension",
    "diccionario", "diccionarios", "dict",
    "set", "sets", "tupla", "tuplas",
    "clase", "clases", "poo", "orientado a objetos",
    "decorador", "decoradores", "generador", "generadores",
    "lambda", "numpy", "pandas", "pytest", "async", "await"
]

def filter_out_of_scope(answer: str, contexto_txt: str) -> tuple[str, bool]:
    """
    Quita párrafos del 'answer' que mencionen términos avanzados
    que NO estén presentes en el contexto. Devuelve (texto_filtrado, se_omito_algo).
    """
    ctx = contexto_txt.lower()

    def para_fuera(p: str) -> bool:
        pl = p.lower()
        for t in ADVANCED_TERMS:
            if t in pl and t not in ctx:
                return True
        return False

    paras = [p for p in answer.split("\n\n") if p.strip()]
    kept = [p for p in paras if not para_fuera(p)]
    omitted = len(kept) != len(paras)
    if not kept:
        return "", True
    return "\n\n".join(kept), omitted

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(body: AskBody):
    ensure_loaded()

    ctx = buscar_contexto(body.pregunta, body.clase)
    if not ctx:
        return JSONResponse({"answer": "Aún no lo vimos en clase.", "fuentes": []})

    prompt = construir_prompt(ctx, body.pregunta)

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )
    answer = completion.choices[0].message.content
    respuesta_html = codeblocks_to_html(answer)
    fuentes = [(prettify_clase(c["clase"]), c["slide"]) for c in ctx]

    return JSONResponse(
        content={
            "answer": answer,
            "respuesta_html": respuesta_html,
            "fuentes": fuentes,
        }
    )

# ======== Localización ========
LIT_NOT_COVERED = {
    "es": "Aún no lo vimos en clase",
    "en": "We haven’t covered this in class yet",
    "pt": "Ainda não vimos isso em aula",
}

NOTE_OMITTED = {
    "es": "**Nota**: Se omitieron sugerencias que exceden el alcance de esta clase.",
    "en": "**Note**: Suggestions beyond this class scope were omitted.",
    "pt": "**Nota**: Sugestões além do escopo desta aula foram omitidas.",
}

LABELS = {
    "es": {
        "context_header": "Contexto (solo esta clase):",
        "exercise_header": "Enunciado del ejercicio:",
        "code_header": "Código del estudiante (Python):",
        "rules_header": "Reglas didácticas IMPORTANTES:",
        "format_header": "Formato de respuesta OBLIGATORIO:",
    },
    "en": {
        "context_header": "Context (this class only):",
        "exercise_header": "Exercise prompt:",
        "code_header": "Student code (Python):",
        "rules_header": "IMPORTANT teaching rules:",
        "format_header": "MANDATORY response format:",
    },
    "pt": {
        "context_header": "Contexto (apenas esta aula):",
        "exercise_header": "Enunciado do exercício:",
        "code_header": "Código do estudante (Python):",
        "rules_header": "Regras didáticas IMPORTANTES:",
        "format_header": "Formato de resposta OBRIGATÓRIO:",
    },
}

INSTRUCTIONS = {
    "es": (
        "Eres un asistente docente del curso. Usa **exclusivamente** el Contexto de la clase indicada.\n"
        f"Si algo no está en el Contexto, responde literalmente: \"{LIT_NOT_COVERED['es']}\".\n"
        "NO introduzcas conceptos que no aparezcan en el Contexto (por ejemplo: listas, diccionarios, POO, "
        "decoradores, generadores, comprehensions, librerías externas, etc.).\n\n"
        f"{LABELS['es']['rules_header']}\n"
        "- NO entregues una solución completa. Sé guía, no resuelvas por el alumno.\n"
        "- Da como máximo 3 orientaciones concretas (viñetas) y, si incluyes código, que sea un **esqueleto incompleto** con TODOs.\n"
        "- Para concatenar texto usa **solo** el operador `+` (lo visto en esta clase). **No uses** `.replace()`, f-strings, ni `.format()`.\n"
        "- Mantente en el alcance de esta clase. Si el alumno requiere algo fuera del Contexto, di: "
        f"\"{LIT_NOT_COVERED['es']}\".\n\n"
        f"{LABELS['es']['format_header']}\n"
        "1) Breve evaluación (1-2 frases) de si cumple el enunciado.\n"
        "2) 3 orientaciones en viñetas, paso a paso, sin resolver todo.\n"
        "3) (Opcional) Un esqueleto de 3-6 líneas con TODOs (sin solución completa), usando concatenación con `+`."
    ),
    "en": (
        "You are a teaching assistant for the course. Use **only** the Context of the specified class.\n"
        f"If something is not in the Context, answer literally: \"{LIT_NOT_COVERED['en']}\".\n"
        "DO NOT introduce concepts that are not in the Context (e.g., lists, dictionaries, OOP, "
        "decorators, generators, comprehensions, external libraries, etc.).\n\n"
        f"{LABELS['en']['rules_header']}\n"
        "- Do NOT provide a complete solution. Be a guide, don’t solve it for the student.\n"
        "- Give at most 3 concrete, step-by-step hints (bullets). If you include code, it must be an **incomplete skeleton** with TODOs.\n"
        "- For string concatenation use **only** the `+` operator (what this class covered). **Do not** use `.replace()`, f-strings, or `.format()`.\n"
        f"- Stay within the scope of this class. If the student needs something outside the Context, say: \"{LIT_NOT_COVERED['en']}\".\n\n"
        f"{LABELS['en']['format_header']}\n"
        "1) Short evaluation (1–2 sentences) of whether the code meets the prompt.\n"
        "2) 3 bullet-point hints, step by step, without fully solving it.\n"
        "3) (Optional) A 3–6 line skeleton with TODOs (not a full solution), using `+` concatenation."
    ),
    "pt": (
        "Você é um assistente docente do curso. Use **exclusivamente** o Contexto da aula indicada.\n"
        f"Se algo não estiver no Contexto, responda literalmente: \"{LIT_NOT_COVERED['pt']}\".\n"
        "NÃO introduza conceitos que não apareçam no Contexto (ex.: listas, dicionários, POO, "
        "decoradores, geradores, comprehensions, bibliotecas externas, etc.).\n\n"
        f"{LABELS['pt']['rules_header']}\n"
        "- NÃO entregue uma solução completa. Seja um guia, não resolva pelo aluno.\n"
        "- Dê no máximo 3 orientações concretas (tópicos) e, se incluir código, que seja um **esqueleto incompleto** com TODOs.\n"
        "- Para concatenar texto, use **apenas** o operador `+` (o conteúdo desta aula). **Não use** `.replace()`, f-strings ou `.format()`.\n"
        f"- Mantenha-se no escopo desta aula. Se o aluno precisar de algo fora do Contexto, diga: \"{LIT_NOT_COVERED['pt']}\".\n\n"
        f"{LABELS['pt']['format_header']}\n"
        "1) Breve avaliação (1–2 frases) se atende ao enunciado.\n"
        "2) 3 orientações em tópicos, passo a passo, sem resolver por completo.\n"
        "3) (Opcional) Um esqueleto de 3–6 linhas com TODOs (sem solução completa), usando concatenação com `+`."
    ),
}

@app.post("/consejo")
def consejo(body: ConsejoBody):
    ensure_loaded()

    # --- 1) Idioma primero (antes del early-return) ---
    idioma = (body.idioma or "es").lower()
    lang = idioma if idioma in ("es", "en", "pt") else "es"

    # --- 0) Si no hay código, responder localizado ---
    if is_blank(body.codigo):
        msgs = {
            "es": (
                "Necesito que pegues tu intento de código para poder ayudarte.\n"
                f"{('Revisá las slides de ' + body.clase + ' y volvé a intentar.') if body.clase else 'Revisá las slides de la clase correspondiente y volvé a intentar.'}"
            ),
            "en": (
                "I need you to paste your code attempt so I can help.\n"
                f"{('Review the slides of ' + body.clase + ' and try again.') if body.clase else 'Review the slides for the corresponding class and try again.'}"
            ),
            "pt": (
                "Preciso que você cole sua tentativa de código para poder ajudar.\n"
                f"{('Revise os slides de ' + body.clase + ' e tente novamente.') if body.clase else 'Revise os slides da aula correspondente e tente novamente.'}"
            ),
        }
        msg = msgs.get(lang, msgs["es"])
        return JSONResponse({"consejo": msg, "consejo_html": f"<pre>{msg}</pre>", "fuentes": []})

    # --- 1) Recuperar contexto específico de la clase ---
    query = f"{body.enunciado}\n\n{body.codigo}"
    ctx = buscar_contexto_por_clase(query=query, clase=body.clase)

    if not ctx:
        msg_by_lang = {
            "es": "Aún no lo vimos en clase o no hay contexto para esta clase.",
            "en": "We haven’t covered this in class yet or there is no context for this class.",
            "pt": "Ainda não vimos isso em aula ou não há contexto para esta aula.",
        }
        return JSONResponse({"consejo": msg_by_lang.get(lang, msg_by_lang["es"]), "fuentes": []})

    contexto_txt = "\n\n".join(
        [f"(Clase: {c['clase']} • Slide: {c['slide']}) {c['text']}" for c in ctx]
    )

    # --- 3) Prompt con restricciones y formato pedagógico (TOTALMENTE LOCALIZADO) ---
    labels = LABELS[lang]
    prompt = (
        INSTRUCTIONS[lang] + "\n\n"
        f"{labels['context_header']}\n"
        f"{contexto_txt}\n\n"
        f"{labels['exercise_header']}\n"
        f"{body.enunciado}\n\n"
        f"{labels['code_header']}\n"
        "```python\n"
        f"{body.codigo}\n"
        "```\n"
    )

    # Si querés también localizar el mensaje de sistema, podés cambiar SYSTEM_MSG por uno por idioma:
    SYSTEM_MSGS = {
        "es": "Actúas como asistente del curso. Responde SOLO con el 'Contexto'. Si la pregunta excede el material, di: 'Aún no lo vimos en clase'. Usa el mismo tono que las diapositivas.",
        "en": "You act as the course assistant. Answer ONLY using the 'Context'. If the question goes beyond the material, say: 'We haven’t covered this in class yet'. Match the tone of the slides.",
        "pt": "Você atua como assistente do curso. Responda APENAS usando o 'Contexto'. Se a pergunta exceder o material, diga: 'Ainda não vimos isso em aula'. Use o mesmo tom dos slides.",
    }
    system_msg = SYSTEM_MSGS.get(lang, SYSTEM_MSGS["es"])

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )
    answer = completion.choices[0].message.content

    # 4) Post-filtro: NO sugerir fuera del alcance de la clase
    filtrado, omitio = filter_out_of_scope(answer, contexto_txt)
    if omitio:
        if not filtrado.strip():
            answer = LIT_NOT_COVERED[lang] + "."
        else:
            answer = filtrado + "\n\n" + NOTE_OMITTED[lang]

    consejo_html = codeblocks_to_html(answer)
    fuentes = [(prettify_clase(c["clase"]), c["slide"]) for c in ctx]

    return JSONResponse(
        {
            "consejo": answer,
            "consejo_html": consejo_html,
            "fuentes": fuentes,
        }
    )
