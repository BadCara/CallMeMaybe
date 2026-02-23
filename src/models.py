from pydantic import BaseModel
from typing import Dict, Any


class ParameterDetail(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDetail]
    returns: ParameterDetail


class PromptTest(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
