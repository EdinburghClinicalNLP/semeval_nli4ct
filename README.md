# SemEval NLI4CT

SemEval 2024 Task 2: Safe Biomedical Natural Language Inference for Clinical Trials

## Resources

- [Competition CodaLab Page](https://codalab.lisn.upsaclay.fr/competitions/16190?secret_key=4863f655-9dd6-43f0-b710-f17cb67af607)
- [Dataset paper](https://arxiv.org/abs/2305.03598)
- [Task Github Repo](https://github.com/ai-systems/Task-2-SemEval-2024/tree/main)

## TL;DR

Clinical trials are conducted to assess the effectiveness and safety of new treatments.
Clinical Trial Reports (CTR), outline the methodology and findings of a clinical trial, and they are used to design and prescribe experimental treatments.
The application of LLMs in critical domains, such as real-world clinical trials, requires further investigations accompanied by the development of novel evaluation methodologies grounded in a more systematic behavioural and causal analyses.

This second iteration is intended to ground NLI4CT in interventional and causal analyses of NLI models (YU et al., 2022), enriching the original NLI4CT dataset with a novel contrast set, developed through the application of a set of interventions on the statements in the NLI4CT test set.

## Solution workflow and Research Questions

![Solution](docs/SemEval_NLI4CT_Solution.png)

Our proposed pipeline is an LLM-based solution which leverages In-Context examples.

### RQ 1: Can LLM perform well in zero-shot setting?

### RQ 1.1: Which LLM perform the best in zero-shot setting?

Challenges to solve:
- How to force the model to output a minimal response to the query? The model tends to give very long answers
  - Explore system message

| Model          | Train Accuracy | Train F1 | Train Precision | Train Recall | Valid Accuracy | Valid F1 | Valid Precision | Valid Recall |
| -------------- | -------------- | -------- | --------------- | ------------ | -------------- | -------- | --------------- | ------------ |
| LLaMA2-7b-chat | 0.49           | 0.4759   | 0.4851          | 0.3259       | 0.5            | 0.4927   | 0.5             | 0.38         |
| Mistral-7b     |  |  |  |  |  |  |  |  |
| MistralLite-7b |  |  |  |  |  |  |  |  |
| GPT-4          |  |  |  |  |  |  |  |  |

:warning: *Note: "Train_\*" performance indicates the performance on the training split, but still in a zero-shot setup* :warning:

### RQ 1.2: Is parameter fine-tuning necessary?

Note: Base model is the best-performing LLM from the previous sub-RQ.

| Model      | Train Accuracy | Train F1 | Train Precision | Train Recall | Valid Accuracy | Valid F1 | Valid Precision | Valid Recall |
| ---------- | -------------- | -------- | --------------- | ------------ | -------------- | -------- | --------------- | ------------ |
| Zero-shot  |  |  |  |  |  |  |  |  |
| 1-shot     |  |  |  |  |  |  |  |  |
| 2-shot     |  |  |  |  |  |  |  |  |
| LoRA       |  |  |  |  |  |  |  |  |

### (Bonus) RQ 1.3: Do pretrained LLMs exhibit hypothesis-only bias?

LLMs may ignore the supplied evidence altogether, and investigation is necessary to understand whether the LLMs predict the same albeit the supplied CTR is different.

### RQ 2: Can LLMs augmented with in-context examples perform better than zero-shot LLMs?

#### RQ 2.1: How to choose the best in-context examples?

In order to achieve the best performance, the pipeline must choose the best in-context examples for specific scenarii.
There are some conditions/heuristics that we can leverage to narrow the search space:

- Clinical trial report sections
- Document length

After the filtering, the retriever should select the best examples.
However, there are several hurdles to consider, such as limited sequence length.
We will experiment with several setup:

- Zero-shot: No in-context examples
- BM25: Naively take top-k examples.
- BM25 + length penalty: Penalise document length with respect to the input length (if the input is long, longer documents should be penalised. If the input is short, longer documents get penalised less). Take top-k examples from the adjusted score, and naively include it as examples to the model.
- Iterative BM25: Naively take 1 top example for $k$ time.
- Iterative BM25 + length penalty: Take 1 top example for $k$ time using the BM25 + length penalty.

The length penalty in BM25 + length penalty can be defined as:
$$
penalty(x, D_i) = \frac{\alpha (avg(|D|)) + avg(|S|) - |x|}{D_i}
$$

where $\alpha$ denotes the number of documents that the pipeline ideally should retrieve, $x$ denotes the statement. In the iterative BM25, we may want to consider the previously retrieved document, such that $x$ denotes the concatenation of retrieved document(s) and the statement. (Discussion: Each model has a different context length limitation. Should this be reflected?)


| Model                           | Train Accuracy | Train F1 | Train Precision | Train Recall | Valid Accuracy | Valid F1 | Valid Precision | Valid Recall |
| ------------------------------- | -------------- | -------- | --------------- | ------------ | -------------- | -------- | --------------- | ------------ |
| Zero-shot                       |  |  |  |  |  |  |  |  |
| BM25                            |  |  |  |  |  |  |  |  |
| BM25 + length penalty           |  |  |  |  |  |  |  |  |
| Iterative BM25                  |  |  |  |  |  |  |  |  |
| Iterative BM25 + length penalty |  |  |  |  |  |  |  |  |

#### RQ 2.2: Is statement sufficient as a query?

Should we also use the CTR section during the retrieval?

#### RQ 2.3: Is domain-adapted dense retriever necessary?

We noticed several statements which require a degree of biomedical knowledge to understand concept synonyms.
Sparse retrievers may not work well for these instances.
Hence, experiments with dense retriever is necessary.

| Model                           | Train Accuracy | Train F1 | Train Precision | Train Recall | Valid Accuracy | Valid F1 | Valid Precision | Valid Recall |
| ------------------------------- | -------------- | -------- | --------------- | ------------ | -------------- | -------- | --------------- | ------------ |
| Zero-shot                       |  |  |  |  |  |  |  |  |
| BM25                            |  |  |  |  |  |  |  |  |
| PubMedBERT (Dense)              |  |  |  |  |  |  |  |  |
| BioLinkBERT (Dense)             |  |  |  |  |  |  |  |  |

### RQ 3: Can LLMs predict in a faithful and consistent manner?

#### RQ 3.1: Can LLMs predict consistently should the input data is lexically altered?

To evaluate the consistency of the LLMs' predictions, we can try to alter the input data while keeping its meaning.
We created a contrastive corpus to evaluate this.
This contrastive corpus is created by replacing entities within the statements with their synonyms.
We utilised `scispacy` pipeline, specifically:
- NER model (`en_ner_bc5cdr_md`) to extract `CHEMICAL` and `DISEASE` entities
- Abbreviation Detector
- Entity linker (UMLS)

We also implemented a naive postprocessing to remove synonyms that contain: `,`, `(`, and `)` which are unlikely to appear in real statements. For instance:
> Original entity: `capecitabine` \
> CUI: `C0671970` \
> Aliases:
>   - Capecitabinum :white_check_mark:
>   - Capecitabine-containing product :white_check_mark:
>   - Capécitabine :white_check_mark:
>   - Capecitabine :white_check_mark:
>   - CAPE :white_check_mark:
>   - Capecitabin :white_check_mark:
>   - Capecitabine (substance) :x:
>   - 5'-Deoxy-5-fluoro-N-[(pentyloxy)carbonyl]-cytidine :x:
>   - pentyl 1-(5-deoxy-β-D-ribofuranosyl)-5-fluoro-1,2-dihydro-2-oxo-4-pyrimidinecarbamate :x:
>   - N(4)-pentyloxycarbonyl-5'-deoxy-5-fluorocytidine :x:

To create the corpus:

```bash
# Create contrastive corpus from the training statements
python scripts/create_contrastive_corpus.py --data_path data/train.json
# Create contrastive corpus from the validation statements
python scripts/create_contrastive_corpus.py --data_path data/dev.json
```

One potential solution to address this consistency is by fine-tuning the model in a contrastive learning framework, forcing the model to encode semantically equivalent statements similarly.

| Model                   | Train Accuracy | Train F1 | Train Precision | Train Recall | Valid Accuracy | Valid F1 | Valid Precision | Valid Recall |
| ----------------------- | -------------- | -------- | --------------- | ------------ | -------------- | -------- | --------------- | ------------ |
| Zero-shot               |  |  |  |  |  |  |  |  |
| Retrieval-augmented ICL |  |  |  |  |  |  |  |  |
| LoRA                    |  |  |  |  |  |  |  |  |
| Contrastive fine-tuning |  |  |  |  |  |  |  |  |

## Task Description

## Research Aims

- "To investigate the consistency of NLI models in their representation of semantic phenomena necessary for complex inference in clinical NLI settings"
- "To investigate the ability of clinical NLI models to perform faithful reasoning, i.e., make correct predictions for the correct reasons."

## Textual Entailment

CTRs can be categorised into 4 sections:
- Eligibility criteria - A set of conditions for patients to be allowed to take part in the clinical trial
- Intervention - Information concerning the type, dosage, frequency, and duration of treatments being studied.
- Results - Number of participants in the trial, outcome measures, units, and the results.
- Adverse events - These are signs and symptoms observed in patients during the clinical trial.

## Intervention targets

- Numerical - LLMs still struggle to consistently apply numerical and quantitative reasoning. As NLI4CT requires this type of inference, we will specifically target the models' numerical and quantitative reasoning abilities.
- Vocabulary and syntax - Acronyms and aliases are significantly more prevalent in clinical texts than general domain texts, and disrupt the performance of clinical NLI models. Additionally, models may experience shortcut learning, relying on syntactic patterns for inference. We target these concepts and patterns with an intervention.
- Semantics - LLMs struggle with complex reasoning tasks when applied to longer premise-hypothesis pairs. We also intervene on the statements to exploit this.
- Notes - The specific type of intervention performed on a statement will not be available at test or training time.

## EDA

### Sequence Length

- With naive concatenation of primary evidence (+ secondary evidence) + statement:
  - Training
    
    ![Training sequence length distribution](docs/train_seq_len_naive_concat.png)
  - Validation
    
    ![Validation sequence length distribution](docs/valid_seq_len_naive_concat.png)

## How to run

Setup the environment:

```
conda env create -f environment.yml
conda activate clinical_peft
```

Run experiments:

```bash
python scripts/train.py experiment=llama2_7b_zeroshot
```
