"""
chunker.py — Chunking sémantique

Principe :
1. Charger tous les documents bruts (data/raw/*.json)
2. Découper chaque document en phrases
3. Calculer l'embedding de chaque phrase (API Mistral, mistral-embed)
4. Mesurer la similarité cosinus entre phrases consécutives
5. Couper là où la similarité chute fortement (= changement de sujet)
6. Garde-fou taille : si un chunk sémantique dépasse MAX_WORDS, le re-découper
   à taille fixe ; si un chunk est trop petit, le fusionner avec le suivant.

Sortie : data/processed/chunks.jsonl
  Chaque ligne = {"text": ..., "source": ..., "chunk_id": ..., "n_sentences": ...}
"""

import os
import re
import json
import glob
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/chunks.jsonl"

EMBED_MODEL = "mistral-embed"
EMBED_BATCH_SIZE = 16          # nb de phrases envoyées par appel API

MAX_WORDS = 500                 # taille max d'un chunk (mots) avant re-découpage forcé
MIN_WORDS = 40                   # en dessous, on fusionne avec le chunk suivant
BREAKPOINT_PERCENTILE = 90       # plus haut = moins de coupures (chunks plus gros)

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# ---------------------------------------------------------------------------
# 1. Chargement des documents bruts
# ---------------------------------------------------------------------------

def _extract_site(item: dict):
    text = item.get("text", "")
    if not text.strip():
        return None
    return {"text": text, "source": item.get("url", "site inconnu")}


def _extract_paper(item: dict):
    text = item.get("abstract", "")
    if not text.strip():
        return None
    title = item.get("title", "")
    arxiv_id = item.get("arxiv_id", "")
    source = (title + " (arXiv:" + arxiv_id + ")") if title else (item.get("pdf_url") or "papier inconnu")
    return {"text": text, "source": source}


def _extract_github(item: dict):
    # Le texte utile est le README ; on essaie plusieurs noms de clé possibles
    # au cas où le scraper les nomme différemment d'un repo à l'autre.
    text = (
        item.get("readme_text")
        or item.get("readme")
        or item.get("readme_content")
        or item.get("content")
        or ""
    )
    description = item.get("description", "")
    full_text = (description + "\n\n" + text).strip() if description else text.strip()

    if not full_text:
        return None

    source = item.get("url") or item.get("repo") or "dépôt GitHub inconnu"
    return {"text": full_text, "source": source}


_EXTRACTORS = {
    "site": _extract_site,
    "paper": _extract_paper,
    "github": _extract_github,
}


def load_raw_documents():
    """
    Charge récursivement tous les fichiers JSON de data/raw/ (y compris dans
    les sous-dossiers github/, papers/, site/), et extrait le texte pertinent
    selon le type de source déclaré dans la clé "source_type" de chaque fichier.
    """
    docs = []
    paths = glob.glob(os.path.join(RAW_DIR, "**", "*.json"), recursive=True)

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]

        for item in items:
            source_type = item.get("source_type", "")
            extractor = _EXTRACTORS.get(source_type)

            if extractor is None:
                print("  [ignoré] " + path + " -- source_type inconnu : '" + str(source_type) + "'")
                continue

            doc = extractor(item)
            if doc is not None:
                docs.append(doc)
            else:
                print("  [vide] " + path + " -- aucun texte exploitable")

    return docs


# ---------------------------------------------------------------------------
# 2. Découpage en phrases
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü0-9])")

MAX_SENTENCE_WORDS = 200  # au-delà, on force un découpage (protège des blocs Markdown/code sans ponctuation)


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    raw_sentences = SENTENCE_SPLIT_RE.split(text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]
    return _cap_sentence_length(raw_sentences)


def _cap_sentence_length(sentences: list[str], max_words: int = MAX_SENTENCE_WORDS) -> list[str]:
    """
    Certains blocs (tableaux Markdown, code, listes denses) n'ont pas de
    ponctuation de fin de phrase et ressortent comme une seule "phrase" géante,
    ce qui peut dépasser la limite de tokens de l'API d'embedding. On les
    redécoupe ici en morceaux de taille raisonnable.
    """
    result = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            result.append(sentence)
        else:
            for i in range(0, len(words), max_words):
                result.append(" ".join(words[i : i + max_words]))
    return result


# ---------------------------------------------------------------------------
# 3. Embeddings des phrases (par batch)
# ---------------------------------------------------------------------------

def embed_sentences(sentences: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(sentences), EMBED_BATCH_SIZE):
        batch = sentences[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# 4. Regroupement sémantique
# ---------------------------------------------------------------------------

def semantic_group(sentences: list[str], embeddings: np.ndarray) -> list[str]:
    """Regroupe des phrases en chunks selon les points de rupture sémantique."""
    if len(sentences) <= 1:
        return sentences

    # Distance = 1 - similarité entre phrases consécutives
    distances = [
        1 - cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    threshold = np.percentile(distances, BREAKPOINT_PERCENTILE)
    breakpoints = {i for i, d in enumerate(distances) if d > threshold}

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if (i - 1) in breakpoints:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# 5. Garde-fous de taille
# ---------------------------------------------------------------------------

def enforce_size_limits(chunks: list[str]) -> list[str]:
    result = []

    for chunk in chunks:
        words = chunk.split()
        if len(words) <= MAX_WORDS:
            result.append(chunk)
        else:
            # re-découpage à taille fixe si le chunk sémantique est trop gros
            for i in range(0, len(words), MAX_WORDS):
                result.append(" ".join(words[i : i + MAX_WORDS]))

    # fusion des chunks trop petits avec le suivant
    merged = []
    buffer = ""
    for chunk in result:
        buffer = (buffer + " " + chunk).strip() if buffer else chunk
        if len(buffer.split()) >= MIN_WORDS:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    return merged


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def run():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    docs = load_raw_documents()
    print(f"{len(docs)} document(s) brut(s) chargé(s) depuis {RAW_DIR}/")

    total_chunks = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for doc_idx, doc in enumerate(docs):
            sentences = split_into_sentences(doc["text"])
            if not sentences:
                continue

            print(f"[{doc_idx + 1}/{len(docs)}] {doc['source']} — {len(sentences)} phrases")

            if len(sentences) <= 2:
                # trop court pour un découpage sémantique utile
                raw_chunks = [" ".join(sentences)]
            else:
                embeddings = embed_sentences(sentences)
                raw_chunks = semantic_group(sentences, embeddings)

            final_chunks = enforce_size_limits(raw_chunks)

            for i, chunk_text in enumerate(final_chunks):
                out.write(
                    json.dumps(
                        {
                            "text": chunk_text,
                            "source": doc["source"],
                            "chunk_id": f"{doc_idx}-{i}",
                            "n_words": len(chunk_text.split()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_chunks += 1

    print(f"\nTerminé : {total_chunks} chunks écrits dans {OUT_PATH}")


if __name__ == "__main__":
    run()