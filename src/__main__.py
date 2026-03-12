import argparse
import json
import sys
from pathlib import Path
from typing import List, Any
from pydantic import ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult
from src.decoder import JSONDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Constrained Decoding")
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    return parser.parse_args()


def load_json(filepath: Path) -> List[Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        return []


def main() -> None:
    args = parse_args()
    raw_funcs = load_json(Path(args.functions_definition))
    functions = []
    for f in raw_funcs:
        try:
            functions.append(FunctionDefinition(**f))
        except ValidationError as e:
            print(f"Skipping invalid function schema: {e}", file=sys.stderr)
    tests = load_json(Path(args.input))
    if not tests or not functions:
        print("Missing required input data. Exiting.", file=sys.stderr)
        sys.exit(1)
    print("Loading model and building caches...")
    model = Small_LLM_Model()
    decoder = JSONDecoder(model, functions)
    results = []
    for test in tests:
        prompt = test.get("prompt", "")
        print(f"Processing: {prompt}")
        try:
            raw_result = decoder.generate_call(prompt)
            validated = FunctionCallResult(**raw_result)
            results.append(validated.model_dump())
        except Exception as e:
            print(f"Handled error for prompt '{prompt}': {e}", file=sys.stderr)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
