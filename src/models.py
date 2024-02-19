import json
import os
from pydantic import BaseModel
from typing import Literal


class ChunkConfig(BaseModel):
	input_dataset: str
	input_splits: list[str]
	input_text_col: str
	output_dataset: str
	strategy: Literal["spacy", "sequence", "constant"]
	split_seq: str | list[str]
	chunk_len: int
	private: bool


class EmbedConfig(BaseModel):
	input_dataset: str
	input_splits: list[str]
	input_text_col: str
	output_dataset: str
	private: bool
	semaphore_bound: int


class WebhookPayloadEvent(BaseModel):
	action: Literal["create", "update", "delete"]
	scope: str


class WebhookPayloadRepo(BaseModel):
	type: Literal["dataset", "model", "space"]
	name: str
	id: str
	private: bool
	headSha: str


class WebhookPayload(BaseModel):
	event: WebhookPayloadEvent
	repo: WebhookPayloadRepo


with open(os.path.join(os.getcwd(), "chunk_config.json")) as c:
	data = json.load(c)
	chunk_config = ChunkConfig.model_validate_json(json.dumps(data))

with open(os.path.join(os.getcwd(), "embed_config.json")) as c:
	data = json.load(c)
	embed_config = EmbedConfig.model_validate_json(json.dumps(data))
