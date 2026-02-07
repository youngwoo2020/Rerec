import pickle
import os
from sklearn.decomposition import PCA

dataset = "beauty"
month = ""
llm_item_emb = pickle.load(open(os.path.join(dataset+"/handled/", month+"itm_emb_np.pkl"), "rb"))

pca = PCA(n_components=64)
pca_item_emb = pca.fit_transform(llm_item_emb)

with open(os.path.join(dataset+"/handled/",month+ "pca64_itm_emb_np.pkl"), "wb") as f:
    pickle.dump(pca_item_emb, f)

pca = PCA(n_components=128)
pca_item_emb = pca.fit_transform(llm_item_emb)

with open(os.path.join(dataset+"/handled/", month+"pca_itm_emb_np.pkl"), "wb") as f:
    pickle.dump(pca_item_emb, f)