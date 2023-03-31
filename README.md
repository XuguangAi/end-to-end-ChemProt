# End-to-End Models for Chemical–Protein Interaction Extraction: Better Tokenization and Span-Based Pipeline Strategies

This repository contains code for our paper: End-to-End Models for Chemical–Protein Interaction Extraction: Better Tokenization and Span-Based Pipeline Strategies.

## Install dependencies
```
pip install -r requirements.txt
```
## Dataset
Download [ChemProt dataset of BioCreative VI](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/). Preprocessed training/development/test datasets are in `chemprot_data/processed_data/json`.

## Run scripts
The code was mainly modified from PURE. Please see more details about arguments in [PURE](https://github.com/princeton-nlp/PURE#Install-dependencies) repository. `PURE_A` to `PURE_E` correspond to 6 models with different relation representations in our paper. Below we show an example running the model with relation representation A.

### Train entity models
```
python 'PURE_A/run_entity.py' \
--do_train --do_eval \
--num_epoch=50 --print_loss_step=50 \
--learning_rate=1e-5 --task_learning_rate=5e-4 \
--train_batch_size=16 \
--eval_batch_size=16 \
--max_span_length=16 \
--context_window=300 \
--task chemprot_5 \
--seed=$seed \
--data_dir "chemprot_data/processed_data/json" \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--output_dir "chemprot_models/chemprot_a/ent_$seed"
```

### Train relation models
```
python 'PURE_A/run_relation.py' \
--task chemprot_5 \
--do_train --train_file "chemprot_data/processed_data/json/train.json" \
--do_eval \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--do_lower_case \
--train_batch_size=16 \
--eval_batch_size=16 \
--learning_rate=2e-5 \
--num_train_epochs=10 \
--context_window=100 \
--max_seq_length=250 \
--seed=$seed \
--entity_output_dir "chemprot_models/chemprot_a/ent_$seed" \
--output_dir "chemprot_models/chemprot_a/rel_$seed"
```

### Inference
```
python 'PURE_A/run_entity.py' \
--do_eval --eval_test \
--max_span_length=16 \
--context_window=300 \
--task chemprot_5 \
--data_dir 'chemprot_data/processed_data/json' \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--output_dir "chemprot_models/chemprot_a/ent_$seed"

python 'PURE_A/run_relation.py' \
--task chemprot_5 \
--do_eval --eval_test \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--do_lower_case \
--context_window=100 \
--max_seq_length=250 \
--entity_output_dir "chemprot_models/chemprot_a/ent_$seed" \
--output_dir "chemprot_models/chemprot_a/rel_$seed/"

python "PURE_A/run_eval.py" --prediction_file "chemprot_models/chemprot_a/rel_$seed/"/predictions.json
```

## Evaluation results

![results](https://github.com/XuguangAi/end-to-end-ChemProt/blob/main/Figs/results.png)







