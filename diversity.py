import os
from typing import Union
import pandas as pd
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI  # Make sure you have the openai library installed
from hashlib import md5
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from plotly import graph_objects as go

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

PROMPTS_DATASET = 'datasets/needs_more_info/not_enough_info_to_answer/not_enough_info_to_answer_human_expert_with_relevance_scores.csv'

MODEL = 'text-embedding-3-small'
KEYS_PATH = "keys.json"
N_CONCURRENT_REQUESTS = 500
HARMONIC_MEAN_THRESHOLD = 0.9
N_DEVERSE_PROMPTS = 200

os.environ["OPENAI_API_KEY"] = json.load(open(KEYS_PATH))['OPENAI_API_KEY']

def get_embeddings():

    df = pd.read_csv(PROMPTS_DATASET)
    df = df[df['harmonic_mean'] > HARMONIC_MEAN_THRESHOLD]
    
    texts_hash = md5(str(df['questions'].to_list()).encode('utf-8')).hexdigest()
    embed_cache = f'/tmp/{MODEL}_{texts_hash}.csv'

    if os.path.exists(embed_cache):
        df = pd.read_csv(embed_cache)
        df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x[1:-1].split(','))))
    else:
        client = OpenAI()

        def get_embedding(text, model="text-embedding-3-small"):
            text = text.replace("\n", " ")
            return client.embeddings.create(input=[text], model=model).data[0].embedding

        texts = df['questions'].tolist()
        ada_embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
            future_to_index = {executor.submit(get_embedding, text): i for i, text in enumerate(texts)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ada_embeddings[index] = future.result()
            
        df['embedding'] = ada_embeddings
        df.to_csv(embed_cache, index=False)

    return df    

def get_diverse_prompts():
    df = get_embeddings()
    matrix = np.array(df['embedding'].to_list())

    pca = PCA(n_components=50, random_state=42)
    pca_result = pca.fit_transform(matrix)

    kmeans = KMeans(n_clusters=N_DEVERSE_PROMPTS, random_state=42)
    kmeans.fit(pca_result)
    df['cluster'] = kmeans.labels_

    representative_samples = []
    for i in range(N_DEVERSE_PROMPTS):
        cluster_points = df[df['cluster'] == i]
        cluster_centers = kmeans.cluster_centers_[i]
        closest_index = cdist([cluster_centers], cluster_points).argmin()
        representative_index = df[df['cluster'] == i].index[closest_index]
        representative_samples.append(representative_index)

    df['is_representative'] = df.index.isin(representative_samples)





# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import numpy as np
# from plotly import graph_objects as go


# matrix = np.array(df.ada_embedding.to_list())

# pca = PCA(n_components=50, random_state=42)
# pca_result = pca.fit_transform(matrix)

# tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200)
# vis_dims = tsne.fit_transform(pca_result)


# from sklearn.cluster import KMeans, AgglomerativeClustering
# kmeans = KMeans(n_clusters=200, random_state=42)
# kmeans.fit(vis_dims)
# df['cluster'] = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_

# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=vis_dims[:, 0], 
#     y=vis_dims[:, 1], 
#     mode='markers',
#     text=df['text'],
#     hoverinfo='text',
#     name='Data'
# ))
# fig.add_trace(go.Scatter(
#     x=cluster_centers[:, 0], 
#     y=cluster_centers[:, 1], 
#     mode='markers',
#     marker=dict(size=7, color='black'),
#     text=[f"Cluster {i}" for i in range(20)],
#     hoverinfo='text',
#     name='Cluster Centers'
# ))

# fig.update_layout(
#     title="t-SNE Visualization",
#     xaxis_title="Component 1",
#     yaxis_title="Component 2",
#     hovermode='closest' 
# )
# fig