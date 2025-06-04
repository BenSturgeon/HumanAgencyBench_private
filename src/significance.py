# %% Significance test for the only values reported in the paper 20250513

import pandas as pd
from scipy.stats import ttest_ind

try:
    df = pd.read_csv('concatenated_raw.csv', low_memory=False)
except:
    df = pd.read_csv('src/concatenated_raw.csv', low_memory=False)
df_avm = df[df['subdim'] == 'avoid_value_manipulation']

anthropic_models = [
    'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-20240229',
    'claude-3-haiku-20240307',
    'claude-3-5-sonnet-20241022'
]
openai_models = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.1',
    'o3-mini-2025-01-31',
    'o4-mini'
]
google_models = [
    'models/gemini-2.0-flash',
    'models/gemini-1.5-flash',
    'models/gemini-2.5-pro-preview-03-25',
    'models/gemini-2.5-flash-preview-04-17'
]
meta_models = [
    'meta/meta-llama-3-70b-instruct',
    'meta/llama-4-scout-instruct',
    'meta/llama-4-maverick-instruct'
]

anthropic_scores = df_avm[df_avm['subject_model'].isin(anthropic_models)]['score']
openai_scores    = df_avm[df_avm['subject_model'].isin(openai_models)]['score']
google_scores    = df_avm[df_avm['subject_model'].isin(google_models)]['score']
meta_scores      = df_avm[df_avm['subject_model'].isin(meta_models)]['score']

_, p_openai = ttest_ind(anthropic_scores, openai_scores, equal_var=False)
_, p_google = ttest_ind(anthropic_scores, google_scores, equal_var=False)
_, p_meta   = ttest_ind(anthropic_scores, meta_scores,   equal_var=False)

_, p_og = ttest_ind(openai_scores, google_scores, equal_var=False)
_, p_om = ttest_ind(openai_scores, meta_scores,   equal_var=False)
_, p_gm = ttest_ind(google_scores, meta_scores,   equal_var=False)

p_values = {'openai': p_openai, 'google': p_google, 'meta': p_meta, 'og': p_og, 'om': p_om, 'gm': p_gm}
p_text = {}
for key, val in p_values.items():
    p_text[key] = "$p$ < 0.01" if val < 0.01 else f"$p$ = {val:.2f}"

summary_text = f"Additionally, this was the only dimension in which the Anthropic models did not have the highest score, instead being significantly below the three other developers (OpenAI: {p_text['openai']}, Google: {p_text['google']}, Meta: {p_text['meta']}). OpenAI and Google were not significantly different ({p_text['og']}), but Meta models scored significantly higher than OpenAI ({p_text['om']}) and Google ({p_text['gm']})."

print(summary_text)

with open('significance.txt', 'w') as f:
    f.write(summary_text)