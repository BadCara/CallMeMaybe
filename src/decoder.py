import json
import numpy as np
from typing import List, Any, Dict
from llm_sdk.llm_sdk import Small_LLM_Model


class JSONDecoder:
    def __init__(self, model: Small_LLM_Model, functions: List[Any]):
        self.model = model
        self.functions = {f.name: f for f in functions}
        vocab_path = self.model.get_path_to_vocab_file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer_data: Dict[str, str] = json.load(f)
            self.vocab: Dict[str, str] = tokenizer_data.get("model",
                                                            {}).get("vocab",
                                                                    {})
            self.id_to_token = {int(v): k for k, v in self.vocab.items()}

    def _is_valid_prefix(self, current_text: str) -> bool:
        target_start = '{"name": "'
        if len(current_text) <= len(target_start):
            return target_start.startswith(current_text)
        if not current_text.startswith(target_start):
            return False
        after_start = current_text[len(target_start):]
        if '"' not in after_start:
            return any(func_name.startswith(after_start) for func_name in self.functions)
        name_end_idx = after_start.index('"')
        generated_name = after_start[:name_end_idx]
        if generated_name not in self.functions:
            return False
        target_transition = '", "parameters": {'
        after_name = after_start[name_end_idx:]
        if len(after_name) <= len(target_transition):
            return target_transition.startswith(after_name)
        if not after_name.startswith(target_transition):
            return False
        after_transition = after_name[len(target_transition):]
        open_braces = after_transition.count('{')
        close_braces = after_transition.count('}')
        if close_braces > open_braces and not current_text.endswith("}"):
            return False
        return True

    def generate_call(self, prompt: str,
                      max_tokens: int = 50) -> Dict[str, Any]:
        system_prompt = f"Convert this request to JSON: '{prompt}'. \n"
        input_ids = self.model.encode(system_prompt)[0].tolist()
        generated_text = ""
        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits_array = np.array(logits, dtype=np.float32)
            for token_id, token_str in self.id_to_token.items():
                clean_token = token_str.replace('Ġ', ' ').replace('Ċ', '\n')
                test_string = generated_text + clean_token
                if not self._is_valid_prefix(test_string):
                    logits_array[token_id] = -np.inf
            next_token_id = int(np.argmax(logits_array))
            next_token_str = self.id_to_token[next_token_id].replace('Ġ', ' ').replace('Ċ', '\n')
            generated_text += next_token_str
            input_ids.append(next_token_id)
            if (generated_text.endswith("}") and
                    generated_text.count("{") == generated_text.count("}")):
                break
        try:
            parsed = json.loads(generated_text)
            return {
                "prompt": prompt,
                "name": parsed.get("name"),
                "parameters": parsed.get("parameters", {})
            }
        except json.JSONDecodeError:
            return {"prompt": prompt, "name": "error", "parameters": {}}

# \x62\x69\x74\x65\x0a
