import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from specInfer.generator import Generator
from specInfer.common import sychronize_time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch



model_path = "google/flan-t5-xl"
model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)
small_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(model.device)

text_summary = """Q: summarize the following text in one or two sentences.
Script: Mao Zedong, Wade-Giles romanization Mao Tse-tung, (born December 26, 1893, Shaoshan, Hunan province, China—died September 9, 1976, Beijing), principal Chinese Marxist theorist, soldier, and statesman who led his country’s communist revolution. Mao was the leader of the Chinese Communist Party (CCP) from 1935 until his death, and he was chairman (chief of state) of the People’s Republic of China from 1949 to 1959 and chairman of the party also until his death.
When China emerged from a half century of revolution as the world’s most populous country and launched itself on a path of economic development and social change, Mao Zedong occupied a critical place in the story of the country’s resurgence. To be sure, he did not play a dominant role throughout the whole struggle. In the early years of the CCP, he was a secondary figure, though by no means a negligible one, and even after the 1940s (except perhaps during the Cultural Revolution) the crucial decisions were not his alone. Nevertheless, looking at the whole period from the foundation of the CCP in 1921 to Mao’s death in 1976, one can fairly regard Mao Zedong as the principal architect of the new China."""

text_summary_cnndm = """Q: summarize the following text in one or two sentences.
Script: LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
A: 
"""

text_sql = """Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer. How many singers do we have?"""

text_gsm8k = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""

text_ende = """Translate English to German: I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period."""

inputs = tokenizer([text_sql], return_tensors="pt").to(model.device)
#print("Input:")
#print(inputs)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
max_new_tokens = 200

# warmup
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]

###################################### Reference Generation ##############################
start = sychronize_time()
ref_generated = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
print(f"Reference Time: {sychronize_time() - start}")
# print(tokenizer.decode(ref_generated), end="\n\n")
print('Reference output: {}'.format(ref_generated))
summary = tokenizer.decode(ref_generated, skip_special_tokens=True)
print("Reference Answer:")
print(f"{summary}")


###################################### Speculative Decoding ##############################
propose_num = 4
generator = Generator(small_model, model, tokenizer, propose_num, True)
start = sychronize_time()
output = generator.generate(input_ids, max_new_tokens, attention_mask=attention_mask)

print("Our generator output:")
print(output.output[0])
print(f"Speculative Decoding Time: {sychronize_time() - start}")
print(f"alpha: {output.alpha_sum / output.sample_steps}, ",
      f"avg # of correct tokens: {output.correct_tokens.shape[-1] / output.propose_steps}")