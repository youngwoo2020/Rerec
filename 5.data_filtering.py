# % cd data
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
import os
import re
from typing import Dict, List, Tuple, Union


true=True
false=False

recent_month = ''
k_core = 3
k_reps = 5
dataset_name = 'beauty'

def parse(path): # for Amazon
    return pd.read_parquet(path)

def parse_meta(path): # for Amazon
    return pd.read_parquet(path)
  


def Amazon(dataset_name, rating_score):

    def pick(df, candidates, default=None):
        for c in candidates:
            if c in df.columns:
                return c
        return default

    data_file = f"handled/{recent_month}{dataset_name}.parquet"

    df = pd.read_parquet(data_file)


    user_col   = pick(df, ["user_id", "reviewerID", "user", "uid"])
    asin_col   = pick(df, ["asin", "item", "product_id"])
    rating_col = pick(df, ["rating", "overall", "stars", "score"])
    time_col   = pick(df, ["ts", "unixReviewTime", "timestamp", "time"])
    title_col  = pick(df, ["title", "summary", "headline"])
    text_col   = pick(df, ["text", "reviewText", "review", "content", "body"])
    parent_col = pick(df, ["parent_asin", "parent", "parentId", "parent_id"])


    missing = [n for n,(c) in {
        "user_id":user_col, "asin":asin_col
    }.items() if c is None]
    if missing:
        raise KeyError(f"Required column(s) missing in {data_file}: {missing}\nAvailable: {list(df.columns)}")


    df_std = pd.DataFrame({
        "user_id":      df[user_col].astype(str),
        "asin":         df[asin_col].astype(str),
        "rating":       df[rating_col] if rating_col else 0,                     # 없으면 0
        "ts":           df[time_col] if time_col else 0,                         # 없으면 0
        "title":        df[title_col].astype(str) if title_col else "",          # 없으면 ""
        "text":         df[text_col].astype(str) if text_col else "",            # 없으면 ""
        "parent_asin":  df[parent_col].astype(str) if parent_col else "",        # 없으면 ""
    })


    df_std["rating"] = pd.to_numeric(df_std["rating"], errors="coerce").fillna(0)
    df_std["ts"] = pd.to_numeric(df_std["ts"], errors="coerce").fillna(0).astype("int64")


    df_std = df_std[df_std["rating"] > float(rating_score)].copy()


    df_llmname = df_std.rename(columns={
        "user_id": "reviewerID",
        "text": "reviewText",
        "rating": "overall",
        "ts": "unixReviewTime",
    })


    datas = list(zip(
        df_llmname["reviewerID"].astype(str),
        df_llmname["asin"].astype(str),
        df_llmname["unixReviewTime"].astype("int64"),
        df_llmname["overall"].astype(int),
        df_llmname["title"].astype(str),
        df_llmname["reviewText"].astype(str),
        df_llmname["parent_asin"].astype(str),
    ))


    asin_map_df = df_std.dropna(subset=["asin"]).copy()
    asin_map_df = asin_map_df[asin_map_df["parent_asin"].astype(str) != ""]
    asin_map_df = asin_map_df.drop_duplicates(subset=["asin"], keep="last")
    asin2parent = dict(zip(asin_map_df["asin"].astype(str), asin_map_df["parent_asin"].astype(str)))

    return datas, asin2parent



def Amazon_meta(dataset_name):

    meta_path = os.path.join("handled", f"meta_{dataset_name}.parquet")

    use_cols = ["parent_asin", "categories", "details"]

    df = pd.read_parquet(meta_path, engine="pyarrow", columns=use_cols)
    df = df.dropna(subset=["parent_asin"]).drop_duplicates(subset=["parent_asin"], keep="last")

    def extract_brand(details):
        if details is None:
            return None
        if isinstance(details, dict):
            v = details.get("brand")
            return v.strip().upper() if isinstance(v, str) and v.strip() else None
        if isinstance(details, (list, tuple)):
            for kv in details:
                if isinstance(kv, (list, tuple)) and len(kv) == 2:
                    k, v = kv
                    if str(k).lower() == "brand" and isinstance(v, str) and v.strip():
                        return v.strip().upper()
                elif isinstance(kv, dict):
                    v = kv.get("brand")
                    if isinstance(v, str) and v.strip():
                        return v.strip().upper()
        return None

    df["brand"] = df["details"].map(extract_brand)

    datas = {}
    for rec in df.itertuples(index=False):
        parent = getattr(rec, "parent_asin")
        datas[parent] = {
            "categories": getattr(rec, "categories", None) if "categories" in df.columns else None,
            "brand": getattr(rec, "brand", None),
        }
    return datas




def add_comma(num):
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]


def get_attribute_Amazon(meta_parent_infos, asin2parent, datamaps, attribute_core):
    def iter_categories(cat_field):
        if cat_field is None:
            return
        if isinstance(cat_field, float) and pd.isna(cat_field):
            return

        if isinstance(cat_field, np.ndarray):
            if cat_field.size == 0:
                return
            cat_field = cat_field.tolist()

        if isinstance(cat_field, (list, tuple)):
            for path in cat_field:
                if path is None:
                    continue

                if isinstance(path, np.ndarray):
                    path = path.tolist()

                if isinstance(path, (list, tuple)):

                    for cat in path[1:]:
                        
                        if isinstance(cat, np.ndarray):
                            cat = cat.tolist()
                        if isinstance(cat, str):
                            s = cat.strip()
                            if s:
                                yield s
                elif isinstance(path, str):
                    s = path.strip()
                    if s:
                        yield s
        elif isinstance(cat_field, str):
            s = cat_field.strip()
            if s:
                yield s



    freq = defaultdict(int)
    for _, info in meta_parent_infos.items():
        b = info.get("brand")
        if b: freq[b] += 1
        for c in iter_categories(info.get("categories")):
            freq[c] += 1


    parent2attrs = {}
    for parent, info in meta_parent_infos.items():
        attrs = []
        b = info.get("brand")
        if b and freq[b] >= attribute_core:
            attrs.append(b)
        for c in iter_categories(info.get("categories")):
            if freq[c] >= attribute_core:
                attrs.append(c)
        parent2attrs[parent] = attrs


    attribute2id, id2attribute = {}, {}
    attributeid2num = defaultdict(int)
    items2attributes = {}
    next_id = 1
    lengths = []

    item2id = datamaps["item2id"]  # str asin -> str item_id

    for asin, iid_str in item2id.items():
        parent = asin2parent.get(asin)
        if not parent:
            continue
        attrs = parent2attrs.get(parent, [])
        lst = []
        for a in attrs:
            if a not in attribute2id:
                attribute2id[a] = next_id
                id2attribute[next_id] = a
                next_id += 1
            aid = attribute2id[a]
            attributeid2num[aid] += 1
            lst.append(aid)
        iid = int(iid_str)
        items2attributes[iid] = lst
        lengths.append(len(lst))

    attr_num = len(attribute2id)
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    print(f"before delete, attribute num:{attr_num}")
    print(f"attributes len, Min:{min(lengths) if lengths else 0}, Max:{max(lengths) if lengths else 0}, Avg.:{avg_len:.4f}")

    datamaps.update({
        "attribute2id": attribute2id,
        "id2attribute": id2attribute,
        "attributeid2num": attributeid2num
    })
    return attr_num, avg_len, datamaps, items2attributes

def get_interaction(datas):  # sort the interactions based on timestamp
    user_seq = {}
    rate_seq = {} 
    title_seq = {}
    review_seq = {}
    time_seq = {}
    parent_seq = {}  

    for data in datas:
        # datas = (user, item, time, rating, title, review, parent_asin)
        user, item, time, rating, title, review, parent = data
        if user in user_seq:
            user_seq[user].append((item, time, rating, title, review, parent))
        else:
            user_seq[user] = [(item, time, rating, title, review, parent)]

    for user, item_time_rating in user_seq.items():
        item_time_rating.sort(key=lambda x: x[1])  # timestamp 기준 정렬
        items, ratings, titles, reviews, times, parents = [], [], [], [], [], []
        for item, t, rating, title, review, parent in item_time_rating:
            items.append(item)
            ratings.append(rating)
            titles.append(title)
            reviews.append(review)
            times.append(t)
            parents.append(parent)
        user_seq[user]   = items
        rate_seq[user]   = ratings
        title_seq[user]  = titles
        review_seq[user] = reviews
        time_seq[user]   = times
        parent_seq[user] = parents  # <<< 추가

    return user_seq, rate_seq, title_seq, review_seq, time_seq, parent_seq  # <<< 반환값 변경



# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

def filter_Kcore(user_items, user_ratings, user_titles, user_reviews, user_times, user_parents,
                 user_core, item_core):

    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)

    while not isKcore:

        for user, num in list(user_count.items()):
            if num < user_core:
                user_items.pop(user, None)
                user_ratings.pop(user, None)
                user_titles.pop(user, None)
                user_reviews.pop(user, None)
                user_times.pop(user, None)
                user_parents.pop(user, None)


        for user in list(user_items.keys()):

            assert len(user_items[user])   == len(user_ratings[user]), f"length mismatch: {user}"
            assert len(user_items[user])   == len(user_titles[user]),  f"length mismatch: {user}"
            assert len(user_items[user])   == len(user_reviews[user]), f"length mismatch: {user}"
            assert len(user_items[user])   == len(user_times[user]),   f"length mismatch: {user}"
            assert len(user_items[user])   == len(user_parents[user]), f"length mismatch: {user}"

            items   = user_items[user]
            ratings = user_ratings[user]
            titles  = user_titles[user]
            reviews = user_reviews[user]
            times   = user_times[user]
            parents = user_parents[user]

            for i in range(len(items) - 1, -1, -1):
                it = items[i]
                if item_count.get(it, 0) < item_core:
                    del items[i]
                    del ratings[i]
                    del titles[i]
                    del reviews[i]
                    del times[i]
                    del parents[i]

            if not items:
                user_items.pop(user, None)
                user_ratings.pop(user, None)
                user_titles.pop(user, None)
                user_reviews.pop(user, None)
                user_times.pop(user, None)
                user_parents.pop(user, None)


        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)

    return user_items, user_ratings, user_titles, user_reviews, user_times, user_parents



def filter_common(user_items, user_t, item_t):

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, item, _ in user_items:
        user_count[user] += 1
        item_count[item] += 1

    User = {}
    for user, item, timestamp in user_items:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User.keys():
            User[user] = []
        User[user].append((item, timestamp))

    new_User = {}
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[1])
        new_hist = [i for i, t in User[userid]]
        new_User[userid] = new_hist

    return new_User



def id_map(user_items, user_ratings, user_titles, user_reviews):
    user2id, item2id, id2user, id2item = {}, {}, {}, {}
    user_id, item_id = 1, 1
    final_data = {}
    final_rating = {} if user_ratings is not None else None
    final_title  = {} if user_titles  is not None else None
    final_review = {} if user_reviews  is not None else None

    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1

        iids  = []
        rlist = [] if user_ratings is not None else None
        tlist = [] if user_titles  is not None else None
        relist = [] if user_reviews is not None else None

        ratings_for_user = user_ratings.get(user, []) if user_ratings is not None else None
        titles_for_user  = user_titles.get(user, [])  if user_titles  is not None else None
        reviews_for_user = user_reviews.get(user, [])  if user_reviews  is not None else None

        for idx, item in enumerate(items):
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])

            if rlist is not None:
                r = ratings_for_user[idx] if idx < len(ratings_for_user) else None
                rlist.append(r)

            if tlist is not None:
                t = titles_for_user[idx] if idx < len(titles_for_user) else None
                tlist.append(t)

            if relist is not None:
                re = reviews_for_user[idx] if idx < len(reviews_for_user) else None
                relist.append(re)

        uid = user2id[user]
        final_data[uid] = iids
        if final_rating is not None:
            final_rating[uid] = rlist

        if final_title is not None:
            final_title[uid] = tlist

        if final_review is not None:
            final_review[uid] = relist

    data_maps = {"user2id": user2id, "item2id": item2id, "id2user": id2user, "id2item": id2item}
    return final_data, final_rating, final_title,final_review, (user_id - 1), (item_id - 1), data_maps

def get_counts(user_items):

    user_count = {}
    item_count = {}

    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count.keys():
                item_count[item] = 1
            else:
                item_count[item] += 1

    return user_count, item_count


def filter_minmum(user_items, user_ratings, user_titles, user_reviews, user_times, user_parents, min_len=3):
    new_user_items = {}
    new_user_ratings = {}
    new_user_titles = {}
    new_user_reviews = {}
    new_user_times = {}
    new_user_parents = {}

    for user, items in user_items.items():
        if len(items) >= min_len:
            new_user_items[user]   = items
            new_user_ratings[user] = user_ratings[user]
            new_user_titles[user]  = user_titles[user]
            new_user_reviews[user] = user_reviews[user]
            new_user_times[user]   = user_times[user]
            new_user_parents[user] = user_parents[user]
    return new_user_items, new_user_ratings, new_user_titles, new_user_reviews, new_user_times, new_user_parents

import ast

def _to_list_safe(x):

    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return [x]  
    return [] if x is None or (isinstance(x, float) and pd.isna(x)) else [str(x)]

def load_rep_sentences(rep_path: str) -> Dict[str, list]:
    ext = os.path.splitext(rep_path)[1].lower()
    df = pd.read_parquet(rep_path)

    assert "asin" in df.columns and "rep_sentences" in df.columns, "rep 파일에 asin, rep_sentences 컬럼이 필요합니다."
    df = df[["asin", "rep_sentences"]].dropna(subset=["asin"]).drop_duplicates(subset=["asin"], keep="last")
    df["asin"] = df["asin"].astype(str)
    df["rep_sentences"] = df["rep_sentences"].map(_to_list_safe)
    return dict(zip(df["asin"], df["rep_sentences"]))



def main(data_name, data_type='Amazon', user_core=k_core, item_core=k_core, rep_path: str = None):
    assert data_type in {'Amazon', 'Yelp', 'New_Amazon'}
    np.random.seed(12345)
    rating_score = 0.0
    attribute_core = 2

    if not rep_path:
        raise ValueError("rep_path를 지정해주세요. (asin, rep_sentences 포함 파켓/파일)")


    asin2reps = load_rep_sentences(rep_path)

    def _has_nonempty(lst):
        return isinstance(lst, list) and any(isinstance(s, str) and s.strip() for s in lst)

    allowed_asins = {a for a, lst in asin2reps.items() if _has_nonempty(lst)}
    print(f"[rep filter] 대표문장≥1개 보유 asin 수: {len(allowed_asins)}")


    datas, asin2parent  = Amazon(data_name, rating_score=rating_score)


    user_items, user_ratings, user_titles, user_reviews, user_times, user_parents = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')


    user_items, user_ratings, user_titles, user_reviews, user_times, user_parents = filter_Kcore(
        user_items, user_ratings, user_titles, user_reviews, user_times, user_parents,
        user_core=user_core, item_core=item_core
    )
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')


    user_items, user_ratings, user_titles, user_reviews, user_times, user_parents = filter_minmum(
        user_items, user_ratings, user_titles, user_reviews, user_times, user_parents, min_len=3
    )

    
    removed_cnt = 0
    for user in list(user_items.keys()):
        items   = user_items[user]
        ratings = user_ratings[user]
        titles  = user_titles[user]
        reviews = user_reviews[user]
        times   = user_times[user]
        parents = user_parents[user]

        for i in range(len(items) - 1, -1, -1):
            if items[i] not in allowed_asins:
                del items[i]; del ratings[i]; del titles[i]; del reviews[i]; del times[i]; del parents[i]
                removed_cnt += 1

        if not items:
            user_items.pop(user, None)
            user_ratings.pop(user, None)
            user_titles.pop(user, None)
            user_reviews.pop(user, None)
            user_times.pop(user, None)
            user_parents.pop(user, None)

    print(f"[rep filter] 대표문장 없는 아이템 제거 수: {removed_cnt}")


    user_items, user_ratings, user_titles, user_reviews, user_times, user_parents = filter_minmum(
        user_items, user_ratings, user_titles, user_reviews, user_times, user_parents, min_len=3
    )

    
    asin2parent = {a: p for a, p in asin2parent.items() if a in allowed_asins}

    rows = []
    for user in user_items.keys():
        items   = user_items[user]
        ratings = user_ratings[user]
        titles  = user_titles[user]
        reviews = user_reviews[user]
        times   = user_times[user]
        parents = user_parents[user]
        assert len(items)==len(ratings)==len(titles)==len(reviews)==len(times)==len(parents)

        for i in range(len(items)):
            rows.append({
                "reviewerID":     str(user),
                "asin":           str(items[i]),
                "unixReviewTime": int(times[i]) if pd.notna(times[i]) else 0,
                "overall":        int(ratings[i]) if pd.notna(ratings[i]) else 0,
                "title":          str(titles[i]) if titles[i] is not None else "",
                "reviewText":     str(reviews[i]) if reviews[i] is not None else "",
                "parent_asin":    str(parents[i]) if parents[i] is not None else "",
            })

    filtered_df = pd.DataFrame(
        rows,
        columns=["reviewerID","asin","unixReviewTime","overall","title","reviewText","parent_asin"]
    )

    filtered_df["rep_sentences"] = filtered_df["asin"].map(
        lambda a: json.dumps(asin2reps.get(str(a), []), ensure_ascii=False)
    )

    out_parquet = os.path.join(f"handled/{recent_month}kcore{k_core}_kreps{k_reps}_{dataset_name}_aff.parquet")
    filtered_df.to_parquet(out_parquet, engine="pyarrow", index=False, compression="zstd")
    print(f"Saved filtered reviews: {out_parquet} (rows={len(filtered_df)})")

    
    user_file =  'handled/'+recent_month+'user_items.json'
    with open(user_file, "w") as f:
        json.dump(user_items, f)

    user_items, user_ratings, user_titles, user_reviews, user_num, item_num, data_maps = id_map(
        user_items, user_ratings, user_titles, user_reviews
    )


    user_count, item_count = get_counts(user_items)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    
    print('Begin extracting meta infos...')
    meta_infos = Amazon_meta(data_name)
    attribute_num, avg_attribute, data_maps, item2attributes = get_attribute_Amazon(
        meta_infos, asin2parent, data_maps, attribute_core
    )

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\\\')

    handled_path = 'handled/'
    os.makedirs(handled_path, exist_ok=True)

    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'inter_seq.txt'), 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'rating_seq.txt'), 'w') as out:
        for user, ratings in user_ratings.items():
            out.write(user + ' ' + ' '.join(map(str, ratings)) + '\n')
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'title_seq.txt'), 'w') as out:
        for user, items in user_titles.items():
            out.write(user + ' ' + '[#^#]'.join(items) + '\n')
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'review_seq.txt'), 'w') as out:
        for user, items in user_reviews.items():
            out.write(user + ' ' + '[#^#]'.join(items) + '\n')
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'item2attributes.json'), 'w') as out:
        json.dump(item2attributes, out)
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'id_map.json'), "w") as f:
        json.dump(data_maps, f)

   
    id2item = data_maps.get("id2item", {})
    items2rep = {}
    for iid_str, asin in id2item.items():
        reps = asin2reps.get(str(asin), [])
        if _has_nonempty(reps):  # 비어있지 않은 것만
            items2rep[int(iid_str)] = reps
    with open(os.path.join(handled_path, f'{recent_month}kcore{k_core}_kreps{k_reps}_'+'items2rep_sentences.json'), 'w', encoding='utf-8') as out:
        json.dump(items2rep, out, ensure_ascii=False)
    print(f"Saved items2rep_sentences: {os.path.join(handled_path, recent_month+'items2rep_sentences.json')}")



if __name__ == "__main__":

  # rep_path에 asin, rep_sentences 컬럼이 있는 파일 경로 전달
  main(dataset_name, data_type="Amazon", user_core=k_core, item_core=k_core,
      rep_path=f"handled/recent_kcore{k_core}_kreps{k_reps}_{dataset_name}_af.parquet")
