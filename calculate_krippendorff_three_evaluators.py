from pathlib import Path
import pandas as pd
import krippendorff

BASE = Path(__file__).parent
MODELS = {
    "gemini_2_5_pro": "gemini_2_5_pro_evaluator",
    "claude_4_sonnet": "claude_4_sonnet_evaluator",
    "o3": "o3_rerun",
}
TASKS = [
    "ask_clarifying_questions",
    "defer_important_decisions",
    "maintain_social_boundaries",
    "avoid_value_manipulation",
    "encourage_learning",
    "correct_misinformation",
]

for task in TASKS:
    # Read each model's scores for this task
    model_dfs = {}
    for col, folder in MODELS.items():
        df = (
            pd.read_csv(BASE / "output" / folder / task / "raw.csv",
                        usecols=["prompt", "score"])
              .assign(score=lambda d: pd.to_numeric(d["score"], errors="coerce"))
              .dropna(subset=["score"])
        )
        model_dfs[col] = df.rename(columns={"score": col})

    # Find common prompts across all models for this task
    common_prompts = set.intersection(*(set(df["prompt"]) for df in model_dfs.values()))
    if not common_prompts:
        print(f"{task}: no common prompts")
        continue

    # For each pair and all three, merge only the relevant dfs
    def get_pivot(cols):
        dfs = [model_dfs[c][model_dfs[c]["prompt"].isin(common_prompts)].set_index("prompt")[c] for c in cols]
        merged = pd.concat(dfs, axis=1)
        return merged.T.to_numpy()

    # All three raters
    pivot_all = get_pivot(list(MODELS))
    alpha_all = krippendorff.alpha(pivot_all, level_of_measurement="interval")

    # Pairwise
    pairs = [
        ("gemini_2_5_pro", "claude_4_sonnet"),
        ("gemini_2_5_pro", "o3"),
        ("claude_4_sonnet", "o3"),
    ]
    alphas = []
    for a, b in pairs:
        pivot = get_pivot([a, b])
        alphas.append(krippendorff.alpha(pivot, level_of_measurement="interval"))

    print(f"\n=== {task} === ({len(common_prompts)} prompts)")
    print(f"all 3 raters:          {alpha_all:6.4f}")
    print(f"gemini vs claude:      {alphas[0]:6.4f}")
    print(f"gemini vs o3:          {alphas[1]:6.4f}")
    print(f"claude vs o3:          {alphas[2]:6.4f}")
