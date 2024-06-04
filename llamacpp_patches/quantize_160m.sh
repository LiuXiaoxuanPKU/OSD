export CUDA_VISIBLE_DEVICES=0

./bin/quantize ../models/code-llama160m/ggml-model-f32.gguf ../models/code-llama160m/ggml-model-f16.gguf f16
./bin/quantize ../models/spider-llama160m/ggml-model-f32.gguf ../models/spider-llama160m/ggml-model-f16.gguf f16
./bin/quantize ../models/gsm8k-llama160m/ggml-model-f32.gguf ../models/gsm8k-llama160m/ggml-model-f16.gguf f16
./bin/quantize ../models/finance-llama160m/ggml-model-f32.gguf ../models/finance-llama160m/ggml-model-f16.gguf f16
