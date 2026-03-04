from pydantic import BaseModel, Field
from typing import Dict, Any


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict)
    returns: Dict[str, str] = Field(default_factory=dict)


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
