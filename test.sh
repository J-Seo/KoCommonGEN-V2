python3 main.py \
--model hf-causal-experimental \
--model_args pretrained="nlpai-lab/KULLM3" \
--task ko_commongen_v2 \
--device cuda:1 \
--num_fewshot 2 \
--batch_size 1 \
--output nlpai-lab/KULLM3 &

#python3 main.py \
#--model hf-causal-experimental \
#--model_args pretrained="mistralai/Mistral-7B-v0.1" \
#--task ko_commongen_v2 \
#--device cuda:1 \
#--num_fewshot 2 \
#--batch_size 1 \
#--output mistralai/Mistral-7B-v0.1 &
#
#python3 main.py \
#--model hf-causal-experimental \
#--model_args pretrained="Qwen/Qwen-7B" \
#--task ko_commongen_v2 \
#--device cuda:2 \
#--num_fewshot 2 \
#--batch_size 1 \
#--output Qwen/Qwen-7B &