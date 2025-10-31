# servidor.py
import os
import json
import re
from typing import List, Optional
from pathlib import Path
import ast
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import unicodedata, re
from groq import Groq
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import time
from collections import OrderedDict
import hashlib, time, re
try:
    import numpy as np
except Exception:
    np = None  # si no está, igual funcionará el match exacto

# Config por defecto (no pisa si ya existen)
CACHE_MAX_ITEMS = globals().get("CACHE_MAX_ITEMS", 300)
SIMILARITY_THRESHOLD = globals().get("SIMILARITY_THRESHOLD", 0.95)

# Estructura LRU (no pisa si ya existe)
_cache = globals().get("_cache") or OrderedDict()
# ---- Cache en memoria para evitar llamadas redundantes al LLM ----
# Estructura: key -> { "respuesta": str, "respuesta_html": str, "fuentes": list, "embed": np.ndarray, "ts": float }
CACHE_MAX_ITEMS = 300           # ajustar según memoria
SIMILARITY_THRESHOLD = 0.95    # umbral coseno para considerar "similar" (0.0-1.0)
_cache = OrderedDict()



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
    allow_origin_regex=r"https://(.+\.vercel\.app)$",
    allow_credentials=False,
    allow_methods=["*"],
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
    force_success:Optional[bool] = False

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
def make_cache_key(idioma: str, clase: Optional[str], enunciado: str, codigo: str) -> str:
    """
    Genera una clave estable que incorpora idioma, clase y enunciado,
    y un hash del código normalizado (espacios colapsados).
    """
    norm_code = re.sub(r"\s+", " ", (codigo or "")).strip()
    hasher = hashlib.sha256()
    hasher.update(norm_code.encode("utf-8"))
    code_hash = hasher.hexdigest()
    key_parts = [idioma or "es", (clase or "").strip(), (enunciado or "").strip(), code_hash]
    return "||".join(key_parts)

def cache_get_similar(key_prefix: str, embed: np.ndarray):
    """
    Busca en la caché entradas con el mismo key_prefix (idioma+clase+enunciado).
    Si encuentra una con similitud >= SIMILARITY_THRESHOLD, devuelve la entry.
    """
    best_key = None
    best_sim = 0.0
    best_item = None
    for k, v in reversed(_cache.items()):  # reversed para preferir entries recientes
        if not k.startswith(key_prefix):
            continue
        cached_embed = v.get("embed")
        if cached_embed is None or embed is None:
            continue
        # coseno
        num = float(np.dot(cached_embed, embed))
        den = float(np.linalg.norm(cached_embed) * np.linalg.norm(embed))
        sim = num / den if den > 0 else 0.0
        if sim > best_sim:
            best_sim = sim
            best_key = k
            best_item = v
            if sim >= SIMILARITY_THRESHOLD:
                break
    if best_item is not None:
        # actualizar LRU: mover al final
        _cache.move_to_end(best_key)
    return best_item, best_sim

def cache_put(key: str, item: dict):
    """Inserta en caché y hace eviction LRU si hace falta."""
    _cache[key] = item
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX_ITEMS:
        _cache.popitem(last=False)  # eliminar el más antiguo
def _normalize_code(code: str) -> str:
    """Colapsa espacios y quita bordes para normalizar el código antes del hash/embedding."""
    return re.sub(r"\s+", " ", (code or "")).strip()

def cache_get_exact(key: str):
    """Devuelve la entrada de caché si existe (y actualiza su posición LRU)."""
    val = _cache.get(key)
    if val is not None:
        _cache.move_to_end(key)
    return val
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

# --- CHEQUEO RÁPIDO: evitar marcar ✅ si hay errores de ejecución evidentes ---
def tiene_error_de_tipo_o_sintaxis(codigo: str) -> bool:
    """
    1) Si hay SyntaxError -> True
    2) Inferencia mínima de tipos:
       - Variable = número -> 'num'
       - Variable = string -> 'str'
       - Detecta operaciones '+' con mezcla num<->str.
    """
    if not codigo:
        return False

    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return True  # error de sintaxis real

    tipos = {}

    # --- 1) Guardar tipos básicos de variables asignadas ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            val = node.value
            tipo_valor = None
            if isinstance(val, ast.Constant):
                if isinstance(val.value, (int, float)):
                    tipo_valor = "num"
                elif isinstance(val.value, str):
                    tipo_valor = "str"
            if tipo_valor:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        tipos[tgt.id] = tipo_valor

    # --- 2) Analizar operaciones de suma ---
    def tipo_expr(expr):
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, (int, float)):
                return "num"
            elif isinstance(expr.value, str):
                return "str"
        elif isinstance(expr, ast.Name):
            return tipos.get(expr.id)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            lt = tipo_expr(node.left)
            rt = tipo_expr(node.right)
            if lt and rt and lt != rt:
                return True  # mezcla de tipos

    return False


def es_actividad_historia(enunciado: str) -> bool:
    """
    Detecta la consigna "Personalizar tu historia" sin acoplarla a un texto exacto,
    usando palabras clave estables.
    """
    if not enunciado:
        return False
    txt = enunciado.lower()
    claves = [
        "personalizar una historia",
        "personalizar tu historia",
        "función input()",
        "concatenación con el operador +",
    ]
    return all(k in txt for k in claves)

def cumple_historia_personalizada(codigo: str) -> Tuple[bool, Dict]:
    """
    Valida 'Personalizar tu historia':
    - Comentarios con historias y palabras [ ] (>=5).
    - 5 input()
    - Usa concatenación con +
    - Al menos 2 print() (título + historia)
    - Hay un print de título (contenga 'Historia' o 'Título')
    """
    codigo = codigo or ""
    lines = codigo.splitlines()

    # 1) Solo comentarios para contar [palabras] en la historia original
    commented = "\n".join(l for l in lines if l.strip().startswith("#"))
    n_brackets = len(re.findall(r"\[[^\]\n]+\]", commented))

    # 2) Contar input()
    n_inputs = len(re.findall(r"\binput\s*\(", codigo))

    # 3) Verificar concatenación con +
    uses_plus = "+" in codigo

    # 4) Verificar impresiones: título + historia
    n_prints = len(re.findall(r"^\s*print\s*\(", codigo, flags=re.M))
    has_title = bool(re.search(
        r'print\s*\(\s*["\']\s*(?:Historia|Título)[^"\']*["\']\s*\)',
        codigo, flags=re.I
    ))

    ok = (n_brackets >= 5 and n_inputs >= 5 and uses_plus and n_prints >= 2 and has_title)
    debug = {
        "n_brackets": n_brackets,
        "n_inputs": n_inputs,
        "uses_plus": uses_plus,
        "n_prints": n_prints,
        "has_title": has_title,
    }
    return ok, debug

CLASS_I18N = {
    "pyth_1000_c01_informacion_extra": {
        "es": "Información Extra",
        "en": "Extra Information",
        "pt": "Informação Extra",
    },
    "pyth_1000_c01_introduccion_sintaxis_02": {
        "es": "Introducción: Sintaxis 02",
        "en": "Introduction: Syntax 02",
        "pt": "Introdução: Sintaxe 02",
    },
    "pyth_1000_c01_introduccion_que_es_python_01": {
        "es": "Introducción: ¿Qué es Python? 01",
        "en": "Introduction: What is Python? 01",
        "pt": "Introdução: O que é Python? 01",
    },
    "pyth_1000_c02_tipos_de_datos": {
        "es": "Tipos de Datos",
        "en": "Data Types",
        "pt": "Tipos de Dados",
    },
    "pyth_1000_c03_modulos": {
        "es": "Módulos",
        "en": "Modules",
        "pt": "Módulos",
    },
    "pyth_1200_c01_funciones": {
        "es": "Funciones",
        "en": "Functions",
        "pt": "Funções",
    },
    "pyth_1200_c02_condicionales": {
        "es": "Condicionales",
        "en": "Conditionals",
        "pt": "Condicionais",
    },
    "pyth_1200_c03_bucles_while": {
        "es": "Bucles While",
        "en": "While Loops",
        "pt": "Laços While",
    },
    "pyth_1300_c01_listas_01": {
        "es": "Listas 01",
        "en": "Lists 01",
        "pt": "Listas 01",
    },
    "pyth_1300_c02_listas_02": {
        "es": "Listas 02",
        "en": "Lists 02",
        "pt": "Listas 02",
    },
    "pyth_1300_c03_bucles_for_in": {
        "es": "Bucles for...in",
        "en": "For...in Loops",
        "pt": "Laços for...in",
    },
    "pyth_1400_c01_diccionarios": {
        "es": "Diccionarios",
        "en": "Dictionaries",
        "pt": "Dicionários",
    },
    "pyth_1400_c02_listas_de_diccionarios": {
        "es": "Listas de Diccionarios",
        "en": "Lists of Dictionaries",
        "pt": "Listas de Dicionários",
    },
}

def _slugify_clase(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFD", name)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quitar tildes
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def translate_class_name(raw_name: str, lang: str) -> str:
    slug = _slugify_clase(raw_name)  # usar el nombre crudo tal cual viene del META
    if slug in CLASS_I18N and lang in ("es", "en", "pt"):
        return CLASS_I18N[slug].get(lang) or CLASS_I18N[slug]["es"]
    return prettify_clase(raw_name)


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
    fuentes = [(translate_class_name(c["clase"], lang), c["slide"]) for c in ctx]

    return JSONResponse(
        content={
            "answer": answer,
            "respuesta_html": respuesta_html,
            "fuentes": fuentes,
        }
    )

# ======== Localización ========
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
        "NO introduzcas conceptos que no aparezcan en el Contexto (listas, diccionarios, POO, etc.).\n\n"
        f"{LABELS['es']['rules_header']}\n"
        "- NO entregues una solución completa. Sé guía, no resuelvas por el alumno.\n"
        "- Da como máximo 3 orientaciones concretas y, si incluyes código, que sea un **esqueleto incompleto** con comentarios o espacios para completar (por ejemplo, usa `___` o `# completar`).\n"
        "- Para concatenar texto usa **solo** `+`. **No uses** `.replace()`, f-strings ni `.format()`.\n"
        "- El material de referencia del enunciado (plantillas, palabras entre [ ]) puede estar en comentarios (#) y debe considerarse válido.\n\n"
        f"{LABELS['es']['format_header']}\n"
        "1) Breve evaluación (1-2 frases) de si cumple.\n"
        "2) 3 orientaciones (viñetas) paso a paso.\n"
        "3) (Opcional) Esqueleto de 3-6 líneas con TODOs (sin solución completa), usando `+`."
    ),
    "en": (
        "You are a teaching assistant. Use **only** the Context from the indicated class.\n"
        f"If something is not in the Context, respond: \"{LIT_NOT_COVERED['en']}\".\n"
        "DO NOT introduce concepts outside the Context (lists, dictionaries, OOP, etc.).\n\n"
        f"{LABELS['en']['rules_header']}\n"
        "- DO NOT provide a complete solution. Be a guide, do not solve for the student.\n"
        "- Give at most 3 concrete guidelines and, if including code, make it an **incomplete skeleton** with comments or placeholders for completion (e.g., use `___` or `# to complete`).\n"
        "- To concatenate text, use **only** `+`. **Do NOT use** `.replace()`, f-strings, or `.format()`.\n"
        "- Reference material from the prompt (templates, words between [ ]) may be in comments (#) and should be considered valid.\n\n"
        f"{LABELS['en']['format_header']}\n"
        "1) Brief evaluation (1-2 sentences) of whether it meets the requirements.\n"
        "2) 3 step-by-step guidelines (bullet points).\n"
        "3) (Optional) 3-6 line skeleton with TODOs (no complete solution), using `+`."
    ),
    "pt": (
        "Você é um assistente de ensino. Use **apenas** o Contexto da aula indicada.\n"
        f"Se algo não estiver no Contexto, responda literalmente: \"{LIT_NOT_COVERED['pt']}\".\n"
        "NÃO introduza conceitos que não apareçam no Contexto (listas, dicionários, POO, etc.).\n\n"
        f"{LABELS['pt']['rules_header']}\n"
        "- NÃO forneça uma solução completa. Seja um guia, não resolva para o aluno.\n"
        "- Dê no máximo 3 orientações concretas e, se incluir código, que seja um **esqueleto incompleto** com comentários ou espaços para completar (por exemplo, use `___` ou `# completar`).\n"
        "- Para concatenar texto, use **apenas** `+`. **Não use** `.replace()`, f-strings ou `.format()`.\n"
        "- O material de referência do enunciado (templates, palavras entre [ ]) pode estar em comentários (#) e deve ser considerado válido.\n\n"
        f"{LABELS['pt']['format_header']}\n"
        "1) Breve avaliação (1-2 frases) se atende aos requisitos.\n"
        "2) 3 orientações passo a passo (itens).\n"
        "3) (Opcional) Esqueleto de 3-6 linhas com TODOs (sem solução completa), usando `+`."
    ),
}

# ======== Respuesta de éxito ========
SUCCESS_REPLY = {
    "es": "✅ Cumple la consigna.",
    "en": "✅ Meets the requirements.",
    "pt": "✅ Atende à consigna.",
}

# ======== Política universal (ampliada) ========
UNIVERSAL_POLICY = {
    "es": (
    "Tu tarea es evaluar el código del estudiante contra el ENUNCIADO.\n"
    "1) Extrae del ENUNCIADO solo los requisitos explícitos (no inventes requisitos, no asumas implícitos).\n"
    "2) Verifica si el código cumple CADA requisito.\n\n"
    "REGLA DE SALIDA:\n"
    "- Si el código cumple correctamente el enunciado y produce la salida esperada, responde exactamente:\n"
    "  ✅ Cumple la consigna.\n"
    "  (No agregues consejos ni la palabra TODOs en este caso.)\n"
    "- No repitas lo que ya está bien. No agregues mejoras de estilo si cumple.\n"
    "- Si el ENUNCIADO es ambiguo o falta información, dilo claramente.\n"
    "- Considera válidos los elementos de referencia presentes en líneas comentadas (#). "
    "Si el ENUNCIADO pide marcar/copiar/incluir texto de referencia (p. ej., palabras entre [ ] o plantillas), "
    "acéptalo cuando aparezca en comentarios y no exijas que se imprima en la salida.\n"
    ),
    "en": (
    "Your task is to evaluate the student's code against the PROMPT.\n"
    "1) Extract only the explicit requirements from the PROMPT (do not invent requirements, do not assume implicit ones).\n"
    "2) Check if the code meets EACH requirement.\n\n"
    "OUTPUT RULE:\n"
    "- If the code correctly meets the prompt and produces the expected output, respond exactly:\n"
    "  ✅ Meets the requirements.\n"
    "  (Do not add advice or the word TODOs in this case.)\n"
    "- Do not repeat what is already correct. Do not add style improvements if it meets the requirements.\n"
    "- If the PROMPT is ambiguous or lacks information, state it clearly.\n"
    "- Consider valid the reference elements present in commented lines (#). "
    "If the PROMPT asks to mark/copy/include reference text (e.g., words between [ ] or templates), "
    "accept it when it appears in comments and do not require it to be printed in the output.\n"
    ),
    "pt": (
    "Sua tarefa é avaliar o código do aluno em relação ao ENUNCIADO.\n"
    "1) Extraia apenas os requisitos explícitos do ENUNCIADO (não invente requisitos, não assuma implícitos).\n"
    "2) Verifique se o código atende CADA requisito.\n\n"
    "REGRAS DE SAÍDA:\n"
    "- Se o código atender corretamente ao enunciado e produzir a saída esperada, responda exatamente:\n"
    "  ✅ Atende à consigna.\n"
    "  (Não adicione conselhos nem a palavra TODOs neste caso.)\n"
    "- Não repita o que já está correto. Não adicione melhorias de estilo se atender aos requisitos.\n"
    "- Se o ENUNCIADO for ambíguo ou faltar informações, diga claramente.\n"
    "- Considere válidos os elementos de referência presentes em linhas comentadas (#). "
    "Se o ENUNCIADO pedir para marcar/copiar/incluir texto de referência (p. ex., palavras entre [ ] ou templates), "
    "aceite quando aparecer em comentários e não exija que seja impresso na saída.\n"
    ),
   
}
    

@app.post("/consejo")
def consejo(body: ConsejoBody):
    ensure_loaded()

    # --- DEBUG opcional ---
    # print("\n===== DEBUG /consejo =====")
    # print("CLASE:", body.clase)
    # print("IDIOMA:", body.idioma)
    # print("ENUNCIADO:\n", body.enunciado)
    # print("CÓDIGO DEL ESTUDIANTE:\n", body.codigo)
    # print("==========================\n")

    # --- 1) Idioma ---
    idioma = (body.idioma or "es").lower()
    lang = idioma if idioma in ("es", "en", "pt") else "es"

    if body.force_success:
        ok_line = SUCCESS_REPLY.get(lang, SUCCESS_REPLY["es"])
        return JSONResponse({
            "consejo": ok_line,
            "consejo_html": f"<p>{ok_line}</p>",
            "fuentes": []
        })
    
    # --- 0) Si no hay código, responder localizado (tu lógica actual) ---
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

    # === PRECHECK SOLO PARA ESTA ACTIVIDAD (Personalizar tu historia) ===
    is_hist = es_actividad_historia(body.enunciado or "")
    precheck_hist_ok = False
    if is_hist:
        ok_local, dbg = cumple_historia_personalizada(body.codigo or "")
        precheck_hist_ok = ok_local
        # print("PRECHECK_LOCAL(historia):", dbg)
        if ok_local:
            # Éxito directo, sin consultar al LLM
            return JSONResponse({
                "consejo": SUCCESS_REPLY.get(lang, SUCCESS_REPLY["es"]),
                "consejo_html": f"<p>{SUCCESS_REPLY.get(lang, SUCCESS_REPLY['es'])}</p>",
                
            })
        # Si NO pasa el precheck, seguimos al LLM, pero NO permitiremos forzar ✅ luego.

    # === PRECHEQUEO DE CACHÉ (para no gastar tokens) ===
    clase_in = (body.clase or "").strip()
    enun_in = (body.enunciado or "").strip()
    codigo_in = (body.codigo or "")

    # Clave exacta (hash del código normalizado) y prefijo contextual para similitud
    cache_key = make_cache_key(lang, clase_in, enun_in, codigo_in)
    key_prefix = f"{lang}||{clase_in}||{enun_in}||"

    # 1) Coincidencia exacta
    _hit = cache_get_exact(cache_key)
    if _hit:
        return JSONResponse({
            "consejo": _hit["respuesta"],
            "consejo_html": _hit.get("respuesta_html", f"<pre>{_hit['respuesta']}</pre>"),
            "fuentes": _hit.get("fuentes", []),
            "cached": True,
            "cache_similarity": 1.0,
        })

    # 2) Coincidencia por similitud (si hay emb_model y numpy)
    embed_vec = None
    try:
        if emb_model is not None and np is not None and not is_blank(codigo_in):
            vec = emb_model.encode([_normalize_code(codigo_in)], normalize_embeddings=True)[0]
            embed_vec = np.array(vec, dtype="float32")
            _hit2, _sim = cache_get_similar(key_prefix, embed_vec)
            if _hit2 and _sim >= SIMILARITY_THRESHOLD:
                return JSONResponse({
                    "consejo": _hit2["respuesta"],
                    "consejo_html": _hit2.get("respuesta_html", f"<pre>{_hit2['respuesta']}</pre>"),
                    "fuentes": _hit2.get("fuentes", []),
                    "cached": True,
                    "cache_similarity": float(_sim),
                })
    except Exception:
        # Si falla el chequeo de embeddings, seguimos normal
        embed_vec = None

    # --- 2) Recuperar contexto específico de la clase (tu lógica actual) ---
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

    labels = LABELS[lang]
    prompt = (
        UNIVERSAL_POLICY.get(lang, UNIVERSAL_POLICY["es"]) + "\n\n"
        + INSTRUCTIONS[lang] + "\n\n"
        f"{labels['context_header']}\n"
        f"{contexto_txt}\n\n"
        f"{labels['exercise_header']}\n"
        f"{body.enunciado}\n\n"
        f"{labels['code_header']}\n"
        "```python\n"
        f"{body.codigo}\n"
        "```\n"
    )

    print(INSTRUCTIONS.get(lang, INSTRUCTIONS["es"]))

    SYSTEM_MSGS = {
        "es": "Actúas como asistente del curso. Responde SOLO con el 'Contexto'. Si la pregunta excede el material, di: 'Aún no lo vimos en clase'. Usa el mismo tono que las diapositivas. Responde SOLO en español.",
        "en": "You act as a course assistant. Respond ONLY with the 'Context'. If the question exceeds the material, say: 'We haven’t covered this in class yet'. Use the same tone as the slides. Respond ONLY in English.",
        "pt": "Você atua como assistente do curso. Responda SOMENTE com o 'Contexto'. Se a pergunta exceder o material, diga: 'Ainda não vimos isso em aula'. Use o mesmo tom dos slides. Responda SOMENTE em português.",
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

    # --- POSTPROCESO UNIVERSAL: forzar ✅ si ya dice que cumple ---
    ok_textos = [
        "cumple con el enunciado",
        "cumple con los requisitos",
        "cumple la consigna",
        "no es necesario agregar nada",
        "ya cumple con lo solicitado",
        "ya realiza correctamente",
        "cumple con el requisito"
    ]

    # 4) Post-filtro: NO sugerir fuera del alcance de la clase
    filtrado, omitio = filter_out_of_scope(answer, contexto_txt)
    if omitio:
        if not filtrado.strip():
            answer = LIT_NOT_COVERED[lang] + "."
        else:
            answer = filtrado + "\n\n" + NOTE_OMITTED[lang]
    else:
        ok_line = SUCCESS_REPLY.get(lang, SUCCESS_REPLY["es"])
        if isinstance(answer, str):
            if lang == "es":
                answer = answer.replace("El código del estudiante", "Tu código").replace("el código del estudiante", "tu código")
            elif lang == "en":
                answer = answer.replace("The student's code", "Your code").replace("the student's code", "your code")
            elif lang == "pt":
                answer = answer.replace("O código do estudante", "Seu código").replace("o código do estudante", "seu código")
            low = answer.lower().strip()
            modelo_dijo_ok = low.startswith("✅") or any(p in low for p in ok_textos)
            # (1) Nunca forzar ✅ en HISTORIA si el precheck NO pasó
            if is_hist and not precheck_hist_ok:
                modelo_dijo_ok = False
            # (2) Para el resto, solo forzar si no hay errores básicos (AST)
            if modelo_dijo_ok and not tiene_error_de_tipo_o_sintaxis(body.codigo or ""):
                answer = ok_line

    consejo_html = codeblocks_to_html(answer)
    fuentes = [(translate_class_name(c["clase"], lang), c["slide"]) for c in ctx]
    # print("Fuentes usadas:", fuentes)

    # === GUARDAR EN CACHÉ para futuros envíos iguales/similares ===
    try:
        if emb_model is not None and np is not None:
            if embed_vec is None:
                vec = emb_model.encode([_normalize_code(codigo_in)], normalize_embeddings=True)[0]
                embed_vec_local = np.array(vec, dtype="float32")
            else:
                embed_vec_local = embed_vec
        else:
            embed_vec_local = None

        cache_put(cache_key, {
            "respuesta": answer,
            "respuesta_html": consejo_html,
            "fuentes": fuentes if isinstance(fuentes, list) else [],
            "embed": embed_vec_local,
            "ts": time.time(),
        })
    except Exception:
        pass

    return JSONResponse(
        {
            "consejo": answer,
            "consejo_html": consejo_html,
            "fuentes": fuentes,
        }
    )
