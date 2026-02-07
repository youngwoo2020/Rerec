from collections import defaultdict
import numpy as np
import pandas as pd
import os

true=True
false=False

recent_month = ''
dataset_name = 'beauty'
k_core = 3


############################
# 1) LOAD DATA
############################

def Amazon(dataset_name, rating_score):
    file_path = f"data/handled/{recent_month}{dataset_name}.parquet"
    df = pd.read_parquet(file_path)

    # 필터
    df = df[df["rating"] > float(rating_score)].copy()

    # rename (reviewText 유지!)
    df = df.rename(columns={
        "user_id": "reviewerID",
        "text": "reviewText",
        "rating": "overall",
        "ts": "unixReviewTime"
    })

    # datas = 9개 필드
    datas = list(zip(
        df["reviewerID"].astype(str),
        df["asin"].astype(str),
        df["unixReviewTime"].fillna(0).astype("int64"),
        df["overall"].fillna(0).astype(int),
        df["title"].astype(str),
        df["reviewText"].astype(str),
        df["parent_asin"].astype(str),
        df["verified_purchase"],
        df["helpful_vote"]
    ))

    # asin → parent_asin mapping
    asin_map = df.dropna(subset=["asin", "parent_asin"]) \
                 .drop_duplicates(subset=["asin"], keep="last")
    asin2parent = dict(zip(
        asin_map["asin"].astype(str),
        asin_map["parent_asin"].astype(str)
    ))

    return datas, asin2parent



############################
# 2) INTERACTION SEQUENCES
############################

def get_interaction(datas):
    """
    datas = (user, item, ts, rating, title, review, parent, verified, helpful)
    """
    user_dict = defaultdict(list)

    # 사용자별 append
    for user, item, ts, rating, title, review, parent, verified, helpful in datas:
        user_dict[user].append((item, ts, rating, title, review, parent, verified, helpful))

    # 정렬 & 시퀀스 분리
    user_items = {}
    user_ratings = {}
    user_titles = {}
    user_reviews = {}
    user_times = {}
    user_parents = {}
    user_verified = {}
    user_helpful = {}

    for user, recs in user_dict.items():
        recs.sort(key=lambda x: x[1])  # timestamp 기준 정렬

        items, ratings, titles, reviews, times, parents, verifieds, helpfuls = [], [], [], [], [], [], [], []

        for item, ts, rating, title, review, parent, vf, hf in recs:
            items.append(item)
            ratings.append(rating)
            titles.append(title)
            reviews.append(review)
            times.append(ts)
            parents.append(parent)
            verifieds.append(vf)
            helpfuls.append(hf)

        user_items[user] = items
        user_ratings[user] = ratings
        user_titles[user] = titles
        user_reviews[user] = reviews
        user_times[user] = times
        user_parents[user] = parents
        user_verified[user] = verifieds
        user_helpful[user] = helpfuls

    return (user_items, user_ratings, user_titles,
            user_reviews, user_times, user_parents,
            user_verified, user_helpful)



############################
# 3) K-CORE FILTERING
############################

def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)

    for user, items in user_items.items():
        user_count[user] = len(items)
        for it in items:
            item_count[it] += 1

    # K-core 만족 여부
    for u, c in user_count.items():
        if c < user_core:
            return user_count, item_count, False
    for it, c in item_count.items():
        if c < item_core:
            return user_count, item_count, False

    return user_count, item_count, True


def filter_Kcore(user_items, user_ratings, user_titles,
                 user_reviews, user_times, user_parents,
                 user_verified, user_helpful,
                 user_core, item_core):

    user_count, item_count, satisfied = check_Kcore(user_items, user_core, item_core)

    while not satisfied:
        # 1) user-core 미만 사용자 제거
        for user, cnt in list(user_count.items()):
            if cnt < user_core:
                for d in (user_items, user_ratings, user_titles,
                          user_reviews, user_times, user_parents,
                          user_verified, user_helpful):
                    d.pop(user, None)

        # 2) item-core 미만 아이템 제거
        for user in list(user_items.keys()):
            items   = user_items[user]
            ratings = user_ratings[user]
            titles  = user_titles[user]
            reviews = user_reviews[user]
            times   = user_times[user]
            parents = user_parents[user]
            verf    = user_verified[user]
            helpf   = user_helpful[user]

            for i in range(len(items) - 1, -1, -1):
                it = items[i]
                if item_count.get(it, 0) < item_core:
                    del items[i]
                    del ratings[i]
                    del titles[i]
                    del reviews[i]
                    del times[i]
                    del parents[i]
                    del verf[i]
                    del helpf[i]

            if not items:
                for d in (user_items, user_ratings, user_titles,
                          user_reviews, user_times, user_parents,
                          user_verified, user_helpful):
                    d.pop(user, None)

        # 재계산
        user_count, item_count, satisfied = check_Kcore(user_items, user_core, item_core)

    return (user_items, user_ratings, user_titles,
            user_reviews, user_times, user_parents,
            user_verified, user_helpful)



############################
# 4) MINIMUM LENGTH FILTER
############################

def filter_minimum(user_items, user_ratings, user_titles,
                   user_reviews, user_times, user_parents,
                   user_verified, user_helpful,
                   min_len=3):

    new_items = {}
    new_ratings = {}
    new_titles = {}
    new_reviews = {}
    new_times = {}
    new_parents = {}
    new_verified = {}
    new_helpful = {}

    for user, items in user_items.items():
        if len(items) >= min_len:
            new_items[user] = items
            new_ratings[user] = user_ratings[user]
            new_titles[user] = user_titles[user]
            new_reviews[user] = user_reviews[user]
            new_times[user] = user_times[user]
            new_parents[user] = user_parents[user]
            new_verified[user] = user_verified[user]
            new_helpful[user] = user_helpful[user]

    return (new_items, new_ratings, new_titles,
            new_reviews, new_times, new_parents,
            new_verified, new_helpful)



############################
# 5) MAIN PIPELINE
############################

def main(data_name, user_core=k_core, item_core=k_core):

    datas, asin2parent = Amazon(data_name, rating_score=0.0)
    print("Raw data loaded from parquet")

    # ① get sequences
    (user_items, user_ratings, user_titles, user_reviews,
     user_times, user_parents, user_verified, user_helpful) = get_interaction(datas)
    print("Interaction sequences built")

    # ② K-core
    (user_items, user_ratings, user_titles, user_reviews,
     user_times, user_parents, user_verified, user_helpful) = filter_Kcore(
        user_items, user_ratings, user_titles, user_reviews,
        user_times, user_parents, user_verified, user_helpful,
        user_core=user_core, item_core=item_core
    )
    print("K-core filtering complete")

    # ③ min length filter
    (user_items, user_ratings, user_titles, user_reviews,
     user_times, user_parents, user_verified, user_helpful) = filter_minimum(
        user_items, user_ratings, user_titles, user_reviews,
        user_times, user_parents, user_verified, user_helpful,
        min_len=3
    )
    print("Minimum length filtering complete")

    # ④ flatten rows
    rows = []
    for user in user_items:
        items   = user_items[user]
        ratings = user_ratings[user]
        titles  = user_titles[user]
        reviews = user_reviews[user]
        times   = user_times[user]
        parents = user_parents[user]
        verfs   = user_verified[user]
        helps   = user_helpful[user]

        for i in range(len(items)):
            rows.append({
                "reviewerID": user,
                "asin": items[i],
                "unixReviewTime": int(times[i]),
                "overall": int(ratings[i]),
                "title": titles[i],
                "reviewText": reviews[i],
                "parent_asin": parents[i],
                "verified_purchase": verfs[i],
                "helpful_vote": helps[i],
            })

    filtered_df = pd.DataFrame(rows)
    out_path = f"data/handled/{recent_month}kcore{k_core}_{dataset_name}_af.parquet"
    filtered_df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)

    print(f"Saved filtered dataset to {out_path}")
    print(f"Final rows = {len(filtered_df)}")


############################
# RUN
############################

if __name__ == "__main__":
    main(dataset_name)
