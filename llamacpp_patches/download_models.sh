git lfs clone https://huggingface.co/lmsys/vicuna-7b-v1.3 ../models/vicuna-7b-v1.3
python3 ../convert.py ../models/vicuna-7b-v1.3

git lfs clone https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3 ../models/TinyLlama-1.1B-Chat-v0.3
python3 ../convert.py ../models/TinyLlama-1.1B-Chat-v0.3

git lfs clone https://huggingface.co/TaylorAI/Flash-Llama-1B-Zombie-2 ../models/Flash-Llama-1B-Zombie-2
python3 ../convert.py ../models/Flash-Llama-1B-Zombie-2

git lfs clone https://huggingface.co/codellama/CodeLlama-7b-hf ../models/CodeLlama-7b-hf
python3 ../convert.py ../models/CodeLlama-7b-hf

git lfs clone https://huggingface.co/codellama/CodeLlama-34b-hf ../models/CodeLlama-34b-hf
python3 ../convert.py ../models/CodeLlama-34b-hf
