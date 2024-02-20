import json

filename = "data/train_paraphrased_wrong.json"
with open(filename, "r") as f:
    data = json.load(f)

keys = list(data.keys())

# Collect the keys of the root samples that have a "Contradiction" label
root_keys_to_delete = []
for key in keys:
    if "_neg" in key or "_pos" in key:
        continue
    else:
        # if data[key]["Label"] == "Contradiction":
        root_keys_to_delete.append(key)

print(root_keys_to_delete)

neg_keys_to_delete = []
for key in root_keys_to_delete:
    # for each root key, delete the perturbed samples with different a label
    for neg_key in keys:
        if neg_key not in data:
            continue
        else:
            if neg_key.startswith(key):
                if data[neg_key]["Label"] != data[key]["Label"]:
                    neg_keys_to_delete.append(neg_key)
            else:
                continue

for key in neg_keys_to_delete:
    del data[key]

# Save data to json file
with open(filename.replace("_wrong", ""), "w") as f:
    json.dump(data, f, indent=4)


# with open("data/train_paraphrased.json", "r") as f:
#     data = json.load(f)

# with open("data/train.json", "r") as f:
#     real_data = json.load(f)

# for key, sample in real_data.items():
#     assert data[key]["Label"] == sample["Label"], f"{key} Labels do not match"

# keys = list(data.keys())
# # Check which keys that have both pos and neg samples
# root_keys = []
# for key in keys:
#     if "_neg" in key or "_pos" in key:
#         continue
#     else:
#         root_keys.append(key)

# # Per root key, check the number of pos and neg samples
# for key in root_keys:
#     entailment_samples = 0
#     contradiction_samples = 0
#     for sample_key in keys:
#         if sample_key.startswith(key):
#             if data[sample_key]["Label"] == "Entailment":
#                 entailment_samples += 1
#             elif data[sample_key]["Label"] == "Contradiction":
#                 contradiction_samples += 1
#     if data[key]["Label"] == "Contradiction" and entailment_samples > 0:
#         print(f" ========= MAYDAY MAYDAY MAYDAY: {key} is problematic ========= ")
#     # print(
#     #     f"{key} ({data[key]['Label']}): {entailment_samples} entailment samples, {contradiction_samples} contradiction samples"
#     # )

# for key, sample in real_data.items():
#     assert data[key]["Label"] == sample["Label"], f"{key} Labels do not match"
