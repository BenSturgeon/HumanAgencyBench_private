import src.dataset_generation as dg
import ast

# result = dg.generate_dataset(
#     subdimension_type="cites_sources",
#     n_prompts=50
# )

# with open("generate_dataset_test_cites_sources_50.txt", "w") as file:
#     file.write(str(result))

with open("generate_dataset_test_cites_sources_50.txt", "r") as file:
    tuple_str = file.read()

tuple_of_lists = ast.literal_eval(tuple_str)

for element in tuple_of_lists[0]:
    print(element)