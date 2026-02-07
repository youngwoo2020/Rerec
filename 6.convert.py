# % cd data
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


recent_month = ''
k_core = 3
k_reps = 5  

data = {}

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"inter_seq.txt", 'r') as f:
    for line in tqdm(f):
        line_data = line.rstrip().split(' ')
        user_id = line_data[0]
        line_data.pop(0)    # delete user_id
        data[user_id] = line_data

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"inter.txt", 'w') as f:
    for user, item_list in tqdm(data.items()):
        for item in item_list:
            u = int(user)
            i = int(item)
            f.write('%s %s\n' % (u, i))

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"rating_seq.txt", 'r') as f:
    for line in tqdm(f):
        line_data = line.rstrip().split(' ')
        user_id = line_data[0]
        line_data.pop(0)    # delete user_id
        data[user_id] = line_data

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"rating.txt", 'w') as f:
    for user, rating_list in tqdm(data.items()):
        for rating in rating_list:
            u = int(user)
            i = int(rating)
            f.write('%s %s\n' % (u, i))

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"title_seq.txt", 'r') as f:
    for line in tqdm(f):
        user, rest = line.rstrip().split(" ", 1)
        titles = rest.split("[#^#]") if rest else []
        data[user] = titles

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"title.txt", 'w') as f:
    for user, title_list in tqdm(data.items()):
        for title in title_list:
            u = int(user)
            i = str(title)
            f.write('%s %s\n' % (u, i))

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"review_seq.txt", 'r') as f:
    for line in tqdm(f):
        user, rest = line.rstrip().split(" ", 1)
        reviews = rest.split("[#^#]") if rest else []
        data[user] = reviews

with open(f"handled/"+f'{recent_month}kcore{k_core}_kreps{k_reps}_'+"review.txt", 'w') as f:
    for user, review_list in tqdm(data.items()):
        for review in review_list:
            u = int(user)
            i = str(review)
            f.write('%s %s\n' % (u, i))

