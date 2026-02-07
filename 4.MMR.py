# !pip uninstall -y torch torchvision torchaudio
# !pip install -U torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install -U sentence-transformers
# !pip install pysbd

# ==== Imports ====
import re, unicodedata, numpy as np, pandas as pd
from typing import List
from datetime import timedelta
import pysbd
from sentence_transformers import SentenceTransformer

month = ''
k_core = 3
k_reps = 5
dataset = 'beauty'

# ==== Config ====

IN_PATH  = f"data/handled/kcore{k_core}_{dataset}_af.parquet"
OUT_PATH = f"data/handled/recent{month}_kcore{k_core}_kreps{k_reps}_{dataset}_af.parquet"
K_REPS   = k_reps
MIN_TOK  = 4
MAX_TOK  = 100
LAMBDA   = 0.7     # MMR λ
DUP_THR  = 0.90    # cosine duplicate threshold
DIVERGE  = 0.85    # max similarity among selected reps
TAU_DAYS = 180     # recency decay constant (days)

# ==== Sentence Split / Clean ====
SEG = pysbd.Segmenter(language="en", clean=False)
_word = re.compile(r"\w+")
cnt = 0
def clean(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    t = re.sub(r"[^\S\r\n]+", " ", t)                   # collapse whitespace
    t = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", t) # remove control chars (keep \r,\n)
    return t.strip()

def split_en(t: str) -> List[str]:
    return [s.strip() for s in SEG.segment(t) if s.strip()]

def toklen(s: str) -> int:
    return len(_word.findall(s))

def split_long(s: str, max_tokens: int) -> List[str]:
    if toklen(s) <= max_tokens: return [s]
    parts = re.split(r"([;:])", s)
    if toklen("".join(parts)) > max_tokens:
        parts = re.split(r"(,)", s)
    out, buf = [], []
    for p in parts:
        buf.append(p)
        if toklen("".join(buf)) >= max_tokens:
            out.append("".join(buf).strip()); buf = []
    if buf: out.append("".join(buf).strip())
    res = []
    for c in out:
        if toklen(c) > max_tokens:
            w = c.split()
            res += [" ".join(w[:max_tokens]), " ".join(w[max_tokens:])]
        else:
            res.append(c)
    return [x for x in res if x]

# ==== Step1: reviews -> sentences (per item) ====
def reviews_to_sentences(df_item: pd.DataFrame) -> pd.DataFrame:
    global cnt
    cnt +=1
    if cnt % 100 == 0:
      print(cnt,"만큼 진행")
    rows = []
    cache = {}  # text -> List[str] (분할 결과 캐시)
    # 빠른 판단용: 단문/단일문장 휴리스틱
    def _likely_single_sentence(t: str) -> bool:
        # 짧고, 종결부호가 거의 없으면 pysbd 생략
        if len(t) < 220:
            p = t.count('.') + t.count('!') + t.count('?')
            return p <= 1
        return False

    for r in df_item.itertuples(index=False):
        txt = clean(r.reviewText)
        if not txt:
            continue

        if txt in cache:
            sent_list = cache[txt]
        else:
            if _likely_single_sentence(txt):
                sent_list = [txt]  # pysbd 호출 생략
            else:
                sent_list = [s.strip() for s in SEG.segment(txt) if s.strip()]
            cache[txt] = sent_list

        sent_idx = 0
        for s in sent_list:
            # 아주 긴 문장만 분절 (불필요 호출 줄이기)
            if toklen(s) > MAX_TOK:
                pieces = split_long(s, MAX_TOK)
            else:
                pieces = [s]

            for piece in pieces:
                if toklen(piece) < MIN_TOK:
                    continue
                rows.append((r.asin, r.reviewerID, r.unixReviewTime, sent_idx, piece))
                sent_idx += 1

    return pd.DataFrame(rows, columns=["asin","reviewerID","unixReviewTime","sent_idx","sentence"])


# ==== Step2: embed + L2 normalize ====
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_norm(texts: List[str]) -> np.ndarray:
    E = MODEL.encode(
        texts, batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False
    )
    return E.astype(np.float32)

# ==== Step3: dedup by cosine > DUP_THR (keep indices) ====
def dedup_keep(texts: List[str], embs: np.ndarray, thr: float = DUP_THR) -> List[int]:
    keep, K = [], None
    for i, e in enumerate(embs):
        if K is None:
            keep.append(i); K = e[None, :]
        else:
            sims = K @ e
            if (sims <= thr).all():
                keep.append(i); K = np.vstack([K, e])
    return keep

# ==== Step4: recency weights ====
def recency_weights(ts_series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts_series, errors="coerce", utc=True)
    now = pd.Timestamp.utcnow()
    days = (now - ts).dt.total_seconds() / 86400.0
    w = np.exp(-days.fillna(days.max()) / TAU_DAYS)
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    return w.to_numpy(dtype=np.float32)

# ==== Step5: MMR select K sentences ====
def mmr_select(embs: np.ndarray, weights: np.ndarray, k: int, lam: float, diverge: float) -> List[int]:
    if len(embs) == 0: return []
    q = (embs * (weights[:, None] + 1e-6)).sum(axis=0)
    q = q / (np.linalg.norm(q) + 1e-12)                    # guard zero-norm
    sims_q = embs @ q
    selected: List[int] = []
    while len(selected) < min(k, len(embs)):
        if not selected:
            i = int(np.argmax(sims_q)); selected.append(i); continue
        sel = embs[selected]
        max_sim = (embs @ sel.T).max(axis=1)
        scores = lam * sims_q - (1 - lam) * max_sim
        scores[selected] = -1e9
        i = int(np.argmax(scores))
        # diversity guard
        if (embs[i] @ sel.T).max() > diverge:
            scores[i] = -1e9
            if scores.max() <= -1e8: break
            i = int(np.argmax(scores))
        selected.append(i)
    return selected


# ==== Step5: MMR select K sentences ====
def mmr_select(embs: np.ndarray, weights: np.ndarray, k: int, lam: float, diverge: float) -> List[int]:
    if len(embs) == 0: return []
    q = (embs * (weights[:, None] + 1e-6)).sum(axis=0)
    q = q / (np.linalg.norm(q) + 1e-12)                    # guard zero-norm
    sims_q = embs @ q
    selected: List[int] = []
    while len(selected) < min(k, len(embs)):
        if not selected:
            i = int(np.argmax(sims_q)); selected.append(i); continue
        sel = embs[selected]
        max_sim = (embs @ sel.T).max(axis=1)
        scores = lam * sims_q - (1 - lam) * max_sim
        scores[selected] = -1e9
        i = int(np.argmax(scores))
        # diversity guard
        if (embs[i] @ sel.T).max() > diverge:
            scores[i] = -1e9
            if scores.max() <= -1e8: break
            i = int(np.argmax(scores))
        selected.append(i)
    return selected


def run(IN_PATH: str, OUT_PATH: str):
    print(f"**파켓 파일 경로:** {IN_PATH}")

    # 필요한 컬럼만 로드 (helpful_vote 제거)
    df = pd.read_parquet(IN_PATH)[[
        "asin", "reviewerID", "unixReviewTime",
        "reviewText", "verified_purchase"
    ]]

    df["verified_purchase"] = df["verified_purchase"].astype(bool)

    out_rows = []

    # asin 별 처리
    for asin, g in df.groupby("asin"):

        # 리뷰 자체가 없는 아이템 제외
        if g["reviewText"].isna().all() or (g["reviewText"].str.len() == 0).all():
            continue

        # 1) 전체 리뷰에 대해 문장 추출
        sents_df = reviews_to_sentences(g)

        if sents_df.empty:
            out_rows.append((asin, []))
            continue

        texts = sents_df["sentence"].tolist()

        # 2) 문장 임베딩
        embs = embed_norm(texts)

        # 3) 중복 제거
        keep_idx = dedup_keep(texts, embs, DUP_THR)
        if len(keep_idx) == 0:
            out_rows.append((asin, []))
            continue

        texts_k = [texts[i] for i in keep_idx]
        embs_k = embs[keep_idx]

        # recency weight만 반영
        ts_k = sents_df.iloc[keep_idx]["unixReviewTime"]
        w = recency_weights(ts_k)

        # 4) MMR로 대표문장 선택
        idx = mmr_select(embs_k, w, K_REPS, LAMBDA, DIVERGE)
        reps = [texts_k[i] for i in idx]

        # 저장
        out_rows.append((asin, reps))

    # parquet 저장
    pd.DataFrame(out_rows, columns=["asin", "rep_sentences"]).to_parquet(OUT_PATH, index=False)


# ==== Execute ====
if __name__ == "__main__":
    run(IN_PATH, OUT_PATH)
    print("Saved:", OUT_PATH)
