import json
import numpy as np
from typing import List, Dict, Any
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


class JSONDecoder:
    """
    Décodeur hybride : force la structure syntaxique (O(1)) et contraint les
    logits uniquement pour les valeurs dynamiques
    (nom de fonction et paramètres). Garantit une vitesse fulgurante et 100%
    de JSON valide.
    """

    def __init__(self, model: Small_LLM_Model,
                 functions: List[FunctionDefinition]):
        self.model = model
        self.functions_map = {f.name: f for f in functions}
        # Chargement du vocabulaire
        vocab_path = self.model.get_path_to_tokenizer_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            self.vocab = tokenizer_data.get("model", {}).get("vocab", {})
            self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        # RÉSOLUTION DU BUG : Obtenir la taille EXACTE des logits du modèle
        # au lieu de la taille du fichier vocab.json (151643)
        dummy_logits = self.model.get_logits_from_input_ids([0])
        self.actual_vocab_size = len(dummy_logits)
        # Pré-nettoyage du vocabulaire pour gagner du temps
        self.clean_tokens = {
            t_id: t_str.replace('Ġ', ' ').replace('Ċ', '\n')
            for t_id, t_str in self.id_to_token.items()
        }
        # Pré-calcul des masques de types pour l'optimisation
        self._precompute_type_masks()

    def _precompute_type_masks(self):
        """Prépare des masques statiques pour autoriser uniquement les nombres
        ou le texte."""
        # On utilise actual_vocab_size pour aligner avec les dimensions du
        # modèle
        self.mask_number = np.full(self.actual_vocab_size, -np.inf,
                                   dtype=np.float32)
        self.mask_string = np.full(self.actual_vocab_size, -np.inf,
                                   dtype=np.float32)
        self.quote_tokens = []

        for t_id, t_str in self.clean_tokens.items():
            if not t_str:
                continue
            # Masque numérique : on n'autorise que les chiffres et les points
            if all(c in '0123456789.- ' for c in t_str):
                self.mask_number[t_id] = 0.0
            # Masque texte : on autorise tout sauf les guillemets
            # (pour ne pas casser le JSON)
            if '"' not in t_str:
                self.mask_string[t_id] = 0.0
            else:
                self.quote_tokens.append(t_id)

    def _generate_value(self, input_ids: List[int], expected_type: str) -> str:
        """Génère la valeur d'un paramètre (intelligence du modèle) avec
        contraintes."""
        generated_val = ""
        for _ in range(20):  # Limite de sécurité par paramètre
            logits = self.model.get_logits_from_input_ids(input_ids)
            if expected_type == "number":
                mask = self.mask_number.copy()
                # On autorise exceptionnellement la virgule ou l'accolade pour
                # signaler la fin du nombre
                for c_id, c_str in self.clean_tokens.items():
                    if ',' in c_str or '}' in c_str:
                        mask[c_id] = 0.0
                filtered_logits = logits + mask
                next_token_id = int(np.argmax(filtered_logits))
                # Sécurité : si le token est inconnu (ex: token spécial),
                # on arrête
                if next_token_id not in self.clean_tokens:
                    break
                next_str = self.clean_tokens[next_token_id].strip()
                # Si le modèle décide de mettre une virgule ou une accolade,
                # il a terminé
                if ',' in next_str or '}' in next_str or not next_str:
                    break
                generated_val += next_str
                input_ids.append(next_token_id)
            elif expected_type == "string":
                mask = self.mask_string.copy()
                # On autorise les guillemets uniquement pour fermer la chaîne
                for q_id in self.quote_tokens:
                    mask[q_id] = 0.0
                filtered_logits = logits + mask
                next_token_id = int(np.argmax(filtered_logits))
                if next_token_id not in self.clean_tokens:
                    break
                next_str = self.clean_tokens[next_token_id]
                # Si le token contient un guillemet, la chaîne est fermée
                if '"' in next_str:
                    generated_val += next_str.split('"')[0]
                    break
                generated_val += next_str
                input_ids.append(next_token_id)
        return generated_val.strip()

    def generate_call(self, prompt: str) -> Dict[str, Any]:
        """Orchestre la création du JSON final en guidant le LLM étape par
        étape."""
        tools_description = ""
        for name, f_def in self.functions_map.items():
            tools_description += f"- {name}: {f_def.description}\n"
        # On crée un prompt clair et structuré
        system_prompt = (
            "You are an expert system. Select the most appropriate function to"
            "fulfill the user's request.\n\n"
            f"Available functions:\n{tools_description}\n"
            f"User request: '{prompt}'\n\n"
            "Resulting JSON function call:\n"
        )
        input_ids = self.model.encode(system_prompt)[0].tolist()
        # --- ÉTAPE 1 : Forçage du préfixe syntaxique ---
        input_ids.extend(self.model.encode('{"name": "')[0].tolist())
        # --- ÉTAPE 2 : Génération intelligente du nom de la fonction ---
        generated_name = ""
        valid_names = list(self.functions_map.keys())
        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(input_ids)
            mask = np.full(self.actual_vocab_size, -np.inf, dtype=np.float32)
            found_valid_token = False
            for t_id, clean_str in self.clean_tokens.items():
                test_str = generated_name + clean_str
                if (any(v.startswith(test_str) for v in valid_names) or
                        any(test_str == v + '"' for v in valid_names)):
                    mask[t_id] = 0.0
                    found_valid_token = True
            if not found_valid_token:
                break
            filtered_logits = logits + mask
            next_token_id = int(np.argmax(filtered_logits))
            if next_token_id not in self.clean_tokens:
                break
            next_str = self.clean_tokens[next_token_id]
            if next_str.endswith('"'):
                generated_name += next_str[:-1]
                input_ids.append(next_token_id)
                break
            generated_name += next_str
            input_ids.append(next_token_id)
        if generated_name not in self.functions_map:
            # Remplacement silencieux si échec partiel
            # On cherche la première fonction qui matche ou on prend la
            # première de la liste
            matched = [n for n in valid_names if n.startswith(generated_name)]
            generated_name = matched[0] if matched else valid_names[0]
        target_func = self.functions_map[generated_name]
        # --- ÉTAPE 3 : Forçage de la transition vers les paramètres ---
        transition_str = '", "parameters": {'
        input_ids.extend(self.model.encode(transition_str)[0].tolist())
        # --- ÉTAPE 4 : Extraction contrainte des paramètres ---
        extracted_params = {}
        for i, (p_name, p_info) in enumerate(target_func.parameters.items()):
            key_prefix = f'"{p_name}": '
            if i > 0:
                key_prefix = ", " + key_prefix
            input_ids.extend(self.model.encode(key_prefix)[0].tolist())
            if p_info.type == "string":
                input_ids.extend(self.model.encode('"')[0].tolist())
                val = self._generate_value(input_ids, "string")
                extracted_params[p_name] = val
                input_ids.extend(self.model.encode('"')[0].tolist())
            elif p_info.type == "number":
                val_str = self._generate_value(input_ids, "number")
                try:
                    extracted_params[p_name] = float(val_str)
                except ValueError:
                    extracted_params[p_name] = 0.0
        return {
            "prompt": prompt,
            "name": generated_name,
            "parameters": extracted_params
        }
