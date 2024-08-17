from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.decomposition import PCA
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.utils import hash_cache

N_CONCURRENT_REQUESTS = 200

def get_embeddings(prompts: List[str], model, use_cache, refresh_cache) -> List[List[float]]:


    client = OpenAI()

    @hash_cache()
    def get_embedding(prompt):
        prompt = prompt.replace("\n", " ")
        return client.embeddings.create(input=[prompt], model=model).data[0].embedding

    embeddings = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {executor.submit(
            get_embedding, 
            prompt=prompt,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc='Getting embeddings'):
            index = future_to_index[future]
            embeddings[index] = future.result()
        
    return embeddings 


def evaluate_prompt_diversity(prompts, model, n_diverse_prompts, use_cache, refresh_cache, problem_type):  # problem_type is accepted just to keep the kwargs consistent with the other functions

    if len(prompts) < 50:
        raise ValueError("The number of prompts should be at least 50 for pca. It's likely that the hmean threshold is to strict.")
    
    embeddings = get_embeddings(prompts, model, use_cache, refresh_cache)

    matrix = np.array(embeddings)

    pca = PCA(n_components=50, random_state=42)
    pca_result = pca.fit_transform(matrix)

    kmeans = KMeans(n_clusters=n_diverse_prompts, random_state=42)
    kmeans.fit(pca_result)
    clusters = kmeans.labels_

    representative_samples = []
    for i in range(n_diverse_prompts):
        cluster_points = pca_result[clusters == i]
        cluster_center = kmeans.cluster_centers_[i]
        closest_index = cdist([cluster_center], cluster_points).argmin()
        original_index = np.where(clusters == i)[0][closest_index]
        representative_samples.append(original_index)

    is_representative = [i in representative_samples for i in range(len(prompts))]

    return embeddings, pca_result.tolist(), clusters.tolist(), representative_samples, is_representative
