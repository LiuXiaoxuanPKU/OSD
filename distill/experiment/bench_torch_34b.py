import torch
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import time
import traceback as tb
import numpy as np

def time_evaluation(origin, compiled, input, forward=None, exp_name: str = '', warmup_time: int = 5) -> None:
    torch.cuda.synchronize()
    s_t = time.time()
    forward(origin, input) if forward else origin(input)
    torch.cuda.synchronize()
    start_t1 = time.time() - s_t
    print(f"Normal firstly used time:{start_t1}s")

    t_1_total = []
    repeat = 10
    for i in range(repeat):
        torch.cuda.synchronize()
        s_t = time.time()
        forward(origin, input) if forward else origin(input)
        torch.cuda.synchronize()
        t_1 = time.time() - s_t
        t_1_total.append(t_1)

        # print(f"{i}:\n\tNormal used time:{t_1}s, \n\t"
        #       f"Compiled used time:{t_2}s")

    print(f"{exp_name} first forward runtime: {start_t1} s")
    print(f"{exp_name} successive runs speed: {np.median(t_1_total)}")
    


def forward_pass(model, input):
    with torch.no_grad():
        try:
            model(**input)
        except:
            tb.print_exc()
            
def bench(model_path):
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16).to(device="cuda:0")
    
    print("#################### Reduce Overhead Compile ########################################")
    text = "Hello World!"
    encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:1")
    rd_compiled_model = torch.compile(model, mode="reduce-overhead")
    time_evaluation(model, rd_compiled_model, encoded_input, forward_pass, model_path.split('/')[-1])

if __name__ == "__main__":
    bench('lmsys/vicuna-33b-v1.3')
    bench('codellama/CodeLlama-34b-hf')
    
