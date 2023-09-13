import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specInfer.generator_seq2seq import Seq2SeqGenerator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch

model_path = "google/t5-efficient-xl"
model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)
small_model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-small").to(model.device)

text = """De Rode Duivels moeten hun droom op een Europese titel opbergen. 
De Belgen verloren vrijdag in München in de kwartfinales van een uitgekookt Italië met 2-1.
België kwam via Barella en Insigne op een dubbele achterstand, maar Lukaku gaf nieuwe hoop 
met een strafschop. De gelijkmaker zat er echter niet meer in.
De Italianen begonnen het beste aan de wedstrijd, al bleven grote kansen uit. Even voor het 
kwartier leek Bonucci uit het niets na een vrije trap met de buik de score te openen, 
maar de videoref keurde de goal af voor buitenspel.
De Belgen moesten het evenwicht herstellen. De Bruyne waagde zijn kans met een afstandschot, dat 
Donnarumma geweldig met de vlakke hand uit doel ranselde. Diezelfde Donnarumma moest 
even nadien Lukaku, na een nieuwe tegenaanval, van de 1-0 houden. De match ging goed op en af - 
de Italianen hadden het meeste balbezit, maar de Belgen loerden onder impuls van De Bruyne 
op de counter."""

text_summary = """Nvidia Corporation is an American multinational technology company incorporated in Delaware and based in Santa Clara, California. It is a software and fabless company which designs graphics processing units (GPUs), application programming interface (APIs) for data science and high-performance computing as well as system on a chip units (SoCs) for the mobile computing and automotive market. Nvidia is a dominant supplier of artificial intelligence hardware and software. Its professional line of GPUs are used in workstations for applications in such fields as architecture, engineering and construction, media and entertainment, automotive, scientific research, and manufacturing design.
In addition to GPU manufacturing, Nvidia provides an API called CUDA that allows the creation of massively parallel programs which utilize GPUs. They are deployed in supercomputing sites around the world. More recently, it has moved into the mobile computing market, where it produces Tegra mobile processors for smartphones and tablets as well as vehicle navigation and entertainment systems. Its competitors include AMD, Intel, Qualcomm and AI-accelerator companies such as Graphcore. It also makes AI-powered software for audio and video processing, e.g. Nvidia Maxine.
Nvidia's GPUs are used for edge-to-cloud computing and supercomputers. Nvidia expanded its presence in the gaming industry with its handheld game consoles Shield Portable, Shield Tablet, and Shield TV and its cloud gaming service GeForce Now."""

input_ids = tokenizer(text_summary, return_tensors="pt", padding=True).input_ids.to(model.device)
max_new_tokens = 100

# warmup
print(input_ids.size())
print(model.generate(input_ids, max_new_tokens=max_new_tokens).size())
ref_generated = model.generate(input_ids, max_new_tokens=max_new_tokens, decoder_start_token_id=model.config.pad_token_id, temperature=0.5)[0]

###################################### Reference Generation ##############################
start = sychronize_time()
ref_generated = model.generate(input_ids, max_new_tokens=max_new_tokens, decoder_start_token_id=model.config.pad_token_id, temperature=0.5)[0]
print(f"Reference Time: {sychronize_time() - start}")
# print(tokenizer.decode(ref_generated), end="\n\n")
summary = tokenizer.decode(ref_generated, skip_special_tokens=True)
print(f"Summary: {summary}")


###################################### Speculative Decoding ##############################
propose_num = 4
generator = Seq2SeqGenerator(small_model, model, tokenizer, propose_num)
start = sychronize_time()
output = generator.generate(input_ids, max_new_tokens, temperature=0.7)

print(output.output)
print(f"Speculative Decoding Time: {sychronize_time() - start}")
print(f"alpha: {output.alpha_sum / output.sample_steps}, ",
      f"avg # of correct tokens: {output.correct_tokens.shape[-1] / output.propose_steps}")