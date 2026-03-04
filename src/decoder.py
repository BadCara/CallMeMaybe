import json
from typing import List, Any, Dict
from llm_sdk.llm_sdk import Small_LLM_Model


class JSONDecoder:
    def __init__(self, model: Small_LLM_Model, functions: List[Any]):
        self.model = model
        self.functions = {f['name']: f for f in functions}
        vocab_path = self.model.get_path_to_vocab_file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer_data: Dict[str, str] = json.load(f)
            self.vocab: Dict[str, str] = tokenizer_data.get("model",
                                                            {}).get("vocab",
                                                                    {})
            self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _is_valid_prefix(self, prefix: str) -> bool:
        if not prefix:
            return True
        

    def generate_json(self, prompt: str, max_tokens: int) -> str:

    
# \x62\x69\x74\x65\x0a
