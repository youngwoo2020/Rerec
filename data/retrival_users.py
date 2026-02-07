import os
import pickle
import json
import numpy as np
from collections import defaultdict
import faiss

# ===== 설정 =====
dataset = "beauty"
base = f"./{dataset}/handled"
recent_month = ""
topk = 100

# ===== 1) inter_seq에서 현재 활성 유저 목록/순서 추출 =====
inter_path = os.path.join(base, recent_month + "inter_seq.txt")
User = defaultdict(list)
user_order = []  # inter_seq 등장 순서

with open(inter_path, "r") as f:
    for line in f:
        user, rest = line.rstrip("\n").split(" ", 1)
        if user not in User:
            user_order.append(user)  # 문자열 ID (id_map의 user_id와 동일한 문자열)
        items = rest.split(" ") if rest else []
        User[user].extend(items)

num_users = len(user_order)
print(f"[inter_seq] users={num_users}")

# user_id는 문자열이지만, 보통 id_map.json에서 "1","2",... 형태.
# 임베딩 파일이 user_id-1 순서로 저장되어 있다면 아래처럼 인덱스 매핑 가능.

# ===== 2) id_map에서 user_id 존재/최대값 확인 =====
id_map_path = os.path.join(base, recent_month + "id_map.json")
with open(id_map_path, "r") as f:
    id_map = json.load(f)

id2user = id_map["id2user"]  # {"1": "orig_user_token", ...}
user2id = id_map["user2id"]  # {"orig_user_token": "1", ...}

# inter_seq의 user는 이미 "1","2",... 문자열일 수도, 원문 토큰일 수도 있음.
# inter_seq의 첫 토큰이 순수 정수 문자열인지 확인하고, 아니라면 user2id로 변환.
def looks_like_int_string(s: str) -> bool:
    return s.isdigit()

if not looks_like_int_string(user_order[0]):
    # inter_seq가 원문 토큰이라면 → id로 변환
    user_order_ids = [user2id[u] for u in user_order if u in user2id]
else:
    # 이미 "1","2",... 형태
    user_order_ids = user_order

# 0-based 인덱스 배열(임베딩 행과 정렬용)
user_rows = np.array([int(uid) - 1 for uid in user_order_ids], dtype=np.int64)

# ===== 3) 유저 임베딩 로드 & inter_seq 순서로 정렬 =====
emb_path = os.path.join(base, recent_month + "user_emb_np.pkl")
user_emb_all = pickle.load(open(emb_path, "rb"))
user_emb_all = np.asarray(user_emb_all, dtype=np.float32)

# 안전 확인 및 슬라이스
assert user_rows.max() < user_emb_all.shape[0], \
    f"임베딩 수({user_emb_all.shape[0]}) < 필요한 최대 인덱스({user_rows.max()})"

user_emb = user_emb_all[user_rows]  # (num_users, d)

# 코사인 유사도용 정규화
faiss.normalize_L2(user_emb)

# ===== 4) FAISS 인덱스 (GPU 가능하면 GPU, 아니면 CPU) =====
d = user_emb.shape[1]
index = faiss.IndexFlatIP(d)
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print("[FAISS] GPU index")
except Exception:
    print("[FAISS] CPU index")

index.add(user_emb)
D, I = index.search(user_emb, topk + 1)  # 자기 자신 포함
final_rank = I[:, 1:1 + topk]            # 첫 열(자기 자신) 제거

# 주의: final_rank는 "user_emb 행 인덱스" 기준이므로,
# inter_seq의 user_id 기준으로 바꾸려면 행→user_id 매핑을 적용.
# 행 r → user_id = int(user_order_ids[r])
final_rank_user_ids = np.vectorize(lambda r: int(user_order_ids[r]))(final_rank)

# ===== 5) 저장(원자적) + 유저 순서도 함께 저장 =====
out_dir = base
os.makedirs(out_dir, exist_ok=True)

# (A) 행 인덱스 기반 이웃행렬(소비 코드가 같은 재정렬을 할 경우에 사용)
out_a = os.path.join(out_dir, recent_month + f"sim_user_{topk}.pkl")
tmp_a = out_a + ".tmp"
with open(tmp_a, "wb") as f:
    pickle.dump(final_rank, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_a, out_a)
print("Saved:", out_a, final_rank.shape)

# (B) user_id(1..N) 기준 이웃행렬(일반적으로 이게 쓰기 쉬움)
out_b = os.path.join(out_dir, recent_month + f"sim_users_uid_{topk}.pkl")
tmp_b = out_b + ".tmp"
with open(tmp_b, "wb") as f:
    pickle.dump(final_rank_user_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_b, out_b)
print("Saved:", out_b, final_rank_user_ids.shape)

# (C) 현재 유저 순서를 저장 (소비 코드에서 안전하게 매핑 가능)
order_path = os.path.join(out_dir, recent_month + "user_order_ids.json")
with open(order_path + ".tmp", "w") as f:
    json.dump(user_order_ids, f)
os.replace(order_path + ".tmp", order_path)
print("Saved order:", order_path, f"{len(user_order_ids)} users")
