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

## Research Aims

- "To investigate the consistency of NLI models in their representation of semantic phenomena necessary for complex inference in clinical NLI settings"
- "To investigate the ability of clinical NLI models to perform faithful reasoning, i.e., make correct predictions for the correct reasons."

## Task Overview: Textual Entailment

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

## TODO

- [X] Setup Language Modelling-based data loader
- Try zero shot performance with LLaMA2-7b-chat and Mistral-7b

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