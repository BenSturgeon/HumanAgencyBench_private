import src.dataset_generation as dg

result = dg.generate_dataset(
    subdimension_type="cites_sources",
    n_prompts=50
)

print(result)


