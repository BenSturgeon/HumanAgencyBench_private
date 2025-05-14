from pathlib import Path
import pandas as pd
import krippendorff

for test in ["examples","preamble"]:
    dfs = []
    for i in range(3):
        p = f"{Path(__file__).parent}/results_gpt41_avm/set_{test}/run_{i}/evaluation_results.csv"
        df_run = pd.read_csv(p)
        df_run['run_index'] = i
        dfs.append(df_run)

    df = pd.concat(dfs, ignore_index=True)
    df['score_numeric'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score_numeric'])
    pivot = df.pivot(index='prompt', columns='run_index', values='score_numeric')
    pivot = pivot.T
    alpha = krippendorff.alpha(reliability_data=pivot, level_of_measurement='interval')
    print(f"{test}: {alpha:.4f}")

df_gpt41 = pd.read_csv(f"{Path(__file__).parent}/results_gpt41_avm/set_preamble/run_0/evaluation_results.csv")
df_claude_sonnet = pd.read_csv(f"{Path(__file__).parent}/results_claude_sonnet_avm/set_preamble/run_0/evaluation_results.csv")

df_gpt41['score_numeric'] = pd.to_numeric(df_gpt41['score'], errors='coerce')
df_claude_sonnet['score_numeric'] = pd.to_numeric(df_claude_sonnet['score'], errors='coerce')

df_g = df_gpt41[['prompt','score_numeric']].dropna().rename(columns={'score_numeric':'gpt41'})
df_c = df_claude_sonnet[['prompt','score_numeric']].dropna().rename(columns={'score_numeric':'claude_sonnet'})

df_m = pd.merge(df_g, df_c, on='prompt')
data = [df_m['gpt41'].tolist(), df_m['claude_sonnet'].tolist()]
alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='interval')
print(f"gpt41 vs claude_sonnet: {alpha:.4f}")

