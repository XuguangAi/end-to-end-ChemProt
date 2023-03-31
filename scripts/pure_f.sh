for seed in 50 51 52 53 54; do 

pip install -r 'PURE_F/requirements.txt'

python 'PURE_F/run_entity.py' \
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
--output_dir "chemprot_models/chemprot_f/ent_$seed"

python 'PURE_F/run_relation.py' \
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
--entity_output_dir "chemprot_models/chemprot_f/ent_$seed" \
--output_dir "chemprot_models/chemprot_f/rel_$seed"

python 'PURE_F/run_entity.py' \
--do_eval --eval_test \
--max_span_length=16 \
--context_window=300 \
--task chemprot_5 \
--data_dir 'chemprot_data/processed_data/json' \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--output_dir "chemprot_models/chemprot_f/ent_$seed"

python 'PURE_F/run_relation.py' \
--task chemprot_5 \
--do_eval --eval_test \
--model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
--do_lower_case \
--context_window=100 \
--max_seq_length=250 \
--entity_output_dir "chemprot_models/chemprot_f/ent_$seed" \
--output_dir "chemprot_models/chemprot_f/rel_$seed/"

python "PURE_F/run_eval.py" --prediction_file "chemprot_models/chemprot_f/rel_$seed/"/predictions.json

done;