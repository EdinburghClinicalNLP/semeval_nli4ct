import os
import zipfile

import wandb

api = wandb.Api()


# EXPERIMENTS_TO_COMPLETE = [
#     "0_shot_llama2_7b_chat",
#     "0_shot_llama2_13b_chat",
#     "0_shot_meditron_7b",
#     "0_shot_mistral_7b_instruct",
#     "0_shot_mistrallite_7b",
#     "1_shot_llama2_7b_chat_retriever_bm25",
#     "1_shot_llama2_7b_chat_retriever_bm25_length_penalty",
#     "1_shot_llama2_7b_chat_retriever_bm25_biolinkbert_reranker",
#     "1_shot_llama2_7b_chat_retriever_bm25_pubmedbert_reranker",
#     "1_shot_llama2_7b_chat_retriever_bm25_contriever_reranker",
#     "1_shot_llama2_13b_chat_retriever_bm25",
#     "1_shot_llama2_13b_chat_retriever_bm25_length_penalty",
#     "1_shot_llama2_13b_chat_retriever_bm25_biolinkbert_reranker",
#     "1_shot_llama2_13b_chat_retriever_bm25_pubmedbert_reranker",
#     "1_shot_llama2_13b_chat_retriever_bm25_contriever_reranker",
#     "1_shot_mistral_7b_instruct_retriever_bm25",
#     "1_shot_mistral_7b_instruct_retriever_bm25_length_penalty",
#     "1_shot_mistral_7b_instruct_retriever_bm25_biolinkbert_reranker",
#     "1_shot_mistral_7b_instruct_retriever_bm25_pubmedbert_reranker",
#     "1_shot_mistral_7b_instruct_retriever_bm25_contriever_reranker",
#     "1_shot_mistrallite_7b_retriever_bm25",
#     "1_shot_mistrallite_7b_retriever_bm25_length_penalty",
#     "1_shot_mistrallite_7b_retriever_bm25_biolinkbert_reranker",
#     "1_shot_mistrallite_7b_retriever_bm25_pubmedbert_reranker",
#     "1_shot_mistrallite_7b_retriever_bm25_contriever_reranker",
#     "2_shot_llama2_7b_chat_retriever_bm25",
#     "2_shot_llama2_7b_chat_retriever_bm25_length_penalty",
#     "2_shot_llama2_7b_chat_retriever_bm25_biolinkbert_reranker",
#     "2_shot_llama2_7b_chat_retriever_bm25_pubmedbert_reranker",
#     "2_shot_llama2_7b_chat_retriever_bm25_contriever_reranker",
#     "2_shot_llama2_13b_chat_retriever_bm25",
#     "2_shot_llama2_13b_chat_retriever_bm25_length_penalty",
#     "2_shot_llama2_13b_chat_retriever_bm25_biolinkbert_reranker",
#     "2_shot_llama2_13b_chat_retriever_bm25_pubmedbert_reranker",
#     "2_shot_llama2_13b_chat_retriever_bm25_contriever_reranker",
#     "2_shot_mistral_7b_instruct_retriever_bm25",
#     "2_shot_mistral_7b_instruct_retriever_bm25_length_penalty",
#     "2_shot_mistral_7b_instruct_retriever_bm25_biolinkbert_reranker",
#     "2_shot_mistral_7b_instruct_retriever_bm25_pubmedbert_reranker",
#     "2_shot_mistral_7b_instruct_retriever_bm25_contriever_reranker",
#     "2_shot_mistrallite_7b_retriever_bm25",
#     "2_shot_mistrallite_7b_retriever_bm25_length_penalty",
#     "2_shot_mistrallite_7b_retriever_bm25_biolinkbert_reranker",
#     "2_shot_mistrallite_7b_retriever_bm25_pubmedbert_reranker",
#     "2_shot_mistrallite_7b_retriever_bm25_contriever_reranker",
#     "late_2_shot_llama2_13b_chat_retriever_bm25",
#     "late_2_shot_llama2_7b_chat_retriever_bm25",
#     "late_2_shot_mistral_7b_instruct_retriever_bm25",
#     "late_2_shot_mistrallite_7b_retriever_bm25",
#     "late_2_shot_meditron_7b_retriever_bm25",
#     "late_4_shot_llama2_13b_chat_retriever_bm25",
#     "late_4_shot_llama2_7b_chat_retriever_bm25",
#     "late_4_shot_mistral_7b_instruct_retriever_bm25",
#     "late_4_shot_mistrallite_7b_retriever_bm25",
#     "late_4_shot_meditron_7b_retriever_bm25",
#     "late_6_shot_llama2_13b_chat_retriever_bm25",
#     "late_6_shot_llama2_7b_chat_retriever_bm25",
#     "late_6_shot_mistral_7b_instruct_retriever_bm25",
#     "late_6_shot_mistrallite_7b_retriever_bm25",
#     "late_6_shot_meditron_7b_retriever_bm25",
#     "late_8_shot_llama2_13b_chat_retriever_bm25",
#     "late_8_shot_llama2_7b_chat_retriever_bm25",
#     "late_8_shot_mistral_7b_instruct_retriever_bm25",
#     "late_8_shot_mistrallite_7b_retriever_bm25",
#     "late_8_shot_meditron_7b_retriever_bm25",
#     "late_coupled_4_shot_llama2_13b_chat_retriever_bm25",
#     "late_coupled_4_shot_llama2_7b_chat_retriever_bm25",
#     "late_coupled_4_shot_mistral_7b_instruct_retriever_bm25",
#     "late_coupled_4_shot_mistrallite_7b_retriever_bm25",
#     "late_coupled_4_shot_meditron_7b_retriever_bm25",
#     "late_coupled_6_shot_llama2_13b_chat_retriever_bm25",
#     "late_coupled_6_shot_llama2_7b_chat_retriever_bm25",
#     "late_coupled_6_shot_mistral_7b_instruct_retriever_bm25",
#     "late_coupled_6_shot_mistrallite_7b_retriever_bm25",
#     "late_coupled_6_shot_meditron_7b_retriever_bm25",
#     "late_coupled_8_shot_llama2_13b_chat_retriever_bm25",
#     "late_coupled_8_shot_llama2_7b_chat_retriever_bm25",
#     "late_coupled_8_shot_mistral_7b_instruct_retriever_bm25",
#     "late_coupled_8_shot_mistrallite_7b_retriever_bm25",
#     "late_coupled_8_shot_meditron_7b_retriever_bm25",
#     "cot_0_shot_llama2_13b_chat",
#     "cot_0_shot_llama2_7b_chat",
#     "cot_0_shot_meditron_7b",
#     "cot_0_shot_mistral_7b_instruct",
#     "cot_0_shot_mistrallite_7b",
#     "cot_1_shot_llama2_13b_chat_retriever_bm25",
#     "cot_1_shot_llama2_7b_chat_retriever_bm25",
#     "cot_1_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_1_shot_mistrallite_7b_retriever_bm25",
#     "cot_1_shot_meditron_7b_retriever_bm25",
#     "cot_2_shot_llama2_13b_chat_retriever_bm25",
#     "cot_2_shot_llama2_7b_chat_retriever_bm25",
#     "cot_2_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_2_shot_mistrallite_7b_retriever_bm25",
#     "cot_2_shot_meditron_7b_retriever_bm25",
#     "cot_late_2_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_2_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_2_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_2_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_2_shot_meditron_7b_retriever_bm25",
#     "cot_late_4_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_4_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_4_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_4_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_4_shot_meditron_7b_retriever_bm25",
#     "cot_late_6_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_6_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_6_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_6_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_6_shot_meditron_7b_retriever_bm25",
#     "cot_late_8_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_8_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_8_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_8_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_8_shot_meditron_7b_retriever_bm25",
#     "cot_late_coupled_4_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_coupled_4_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_coupled_4_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_coupled_4_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_coupled_4_shot_meditron_7b_retriever_bm25",
#     "cot_late_coupled_6_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_coupled_6_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_coupled_6_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_coupled_6_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_coupled_6_shot_meditron_7b_retriever_bm25",
#     "cot_late_coupled_8_shot_llama2_13b_chat_retriever_bm25",
#     "cot_late_coupled_8_shot_llama2_7b_chat_retriever_bm25",
#     "cot_late_coupled_8_shot_mistral_7b_instruct_retriever_bm25",
#     "cot_late_coupled_8_shot_mistrallite_7b_retriever_bm25",
#     "cot_late_coupled_8_shot_meditron_7b_retriever_bm25",
#     "fine_tune_llama2_7b_chat",
#     "fine_tune_llama2_13b_chat",
#     "fine_tune_meditron_7b",
#     "fine_tune_mistral_7b_instruct",
#     "fine_tune_mistrallite_7b",
#     "fine_tune_contrastive_llama2_7b_chat",
#     "fine_tune_contrastive_llama2_13b_chat",
#     "fine_tune_contrastive_meditron_7b",
#     "fine_tune_contrastive_mistral_7b_instruct",
#     "fine_tune_contrastive_mistrallite_7b",
# ]

# missing_experiments = EXPERIMENTS_TO_COMPLETE.copy()
# completed_experiments = []

# experiment_names = []

# version = 1
# while True:
#     if version > 200:
#         break
#     else:
#         try:
#             artifact = api.artifact(
#                 f"aryopg/NLI4CT/results:v{version}", type="submission"
#             )
#             logged_by = artifact.logged_by()
#             experiment_name = []
#             for arg in logged_by.metadata["args"]:
#                 if arg.startswith("experiment="):
#                     experiment_name += [
#                         arg.replace("experiment=", "").replace("/", "_")
#                     ]

#                 if arg.startswith("retriever="):
#                     experiment_name += [arg.replace("=", "_").replace("/", "_")]

#             experiment_name = "_".join(experiment_name)

#             if experiment_name in experiment_names:
#                 experiment_name += f"_v{version}"

#             experiment_names += experiment_name

#             if os.path.isdir(f"./artifacts/{experiment_name}"):
#                 print(version)
#             else:
#                 artifact_dir = artifact.download()
#                 print(artifact_dir)

#                 # rename the downloaded basename part of the  artifact_dir to experiment_name
#                 os.rename(artifact_dir, f"./artifacts/{experiment_name}")

#             completed_experiments += [experiment_name]
#             missing_experiments.remove(experiment_name)
#         except:
#             print("Failed to retrieve:", version)

#         version += 1


# print("Completed:", completed_experiments)
# print("Missing:", missing_experiments)


def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, "w") as zip_file:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)


submission_folder = "../semeval_submissions"
for filepath in os.listdir(submission_folder):
    if filepath.startswith(".") or filepath.endswith(".zip"):
        continue

    folder_to_zip = os.path.join(submission_folder, filepath)
    zip_name = f"{filepath}.zip"
    zip_folder(folder_to_zip, zip_name)
