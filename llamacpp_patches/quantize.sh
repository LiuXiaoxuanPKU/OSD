export CUDA_VISIBLE_DEVICES=0

./bin/quantize ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-f32.gguf ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-f16.gguf f16
./bin/quantize ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-f32.gguf ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-q8_0.gguf q8_0
./bin/quantize ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-f32.gguf ../models/TinyLlama-1.1B-Chat-v0.3/ggml-model-q4_0.gguf q4_0

./bin/quantize ../models/Flash-Llama-1B-Zombie-2/ggml-model-f16.gguf ../models/Flash-Llama-1B-Zombie-2/ggml-model-q8_0.gguf q8_0
./bin/quantize ../models/Flash-Llama-1B-Zombie-2/ggml-model-f16.gguf ../models/Flash-Llama-1B-Zombie-2/ggml-model-q4_0.gguf q4_0

./bin/quantize ../models/CodeLlama-7b-hf/ggml-model-f16.gguf ../models/CodeLlama-7b-hf/ggml-model-q8_0.gguf q8_0
./bin/quantize ../models/CodeLlama-7b-hf/ggml-model-f16.gguf ../models/CodeLlama-7b-hf/ggml-model-q4_0.gguf q4_0

./bin/quantize ../models/CodeLlama-34b-hf/ggml-model-f16.gguf ../models/CodeLlama-34b-hf/ggml-model-q8_0.gguf q8_0
./bin/quantize ../models/CodeLlama-34b-hf/ggml-model-f16.gguf ../models/CodeLlama-34b-hf/ggml-model-q4_0.gguf q4_0
