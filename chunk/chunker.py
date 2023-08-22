from typing import List
from fastchat.model import get_conversation_template

class Chunker:
    def __init__(self) -> None:
        pass
    
    def chunk(self, prompt: str) -> List[str]:
        pass
    
class DummyChunker(Chunker):
    def chunk(self, prompt: str) -> List[str]:
        return [prompt]
    
class LongchatChunker(Chunker):
    def __init__(self, chunk_size=8) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        
    def chunk(self, prompt: str) -> List[str]:
        start_mark = "Below is a record of lines I want you to remember."
        assert start_mark in prompt
        prefix = prompt[:prompt.index("\n")]
        over_mark = "Now the record is over." 
        assert over_mark in prompt
        question = prompt[prompt.index(over_mark):]
        lines = prompt[len(prefix) + 2:-(len(question)+2)].split('\n')
        chunk_len = len(lines) // self.chunk_size
        prompts = []
        start = 0
        for _ in range(self.chunk_size - 1):
            prompts.append('\n'.join(lines[start: start + chunk_len]))
            start += chunk_len
        prompts.append('\n'.join(lines[start:]))
        chunked_prompts = [f"{prefix}\n\n{p}\n\n{question}" for p in prompts]
        
        # add vicuna template
        prompt_with_templates = []
        for p in chunked_prompts:
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], p)
            conv.append_message(conv.roles[1], None)
            prompt_with_templates.append(conv.get_prompt())
            
        return prompt_with_templates