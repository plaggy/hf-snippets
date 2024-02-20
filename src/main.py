import asyncio
import logging
import numpy as np
import time
import json
import os
import tempfile
import requests

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from aiohttp import ClientSession
from langchain.text_splitter import SpacyTextSplitter
from datasets import Dataset, load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.models import chunk_config, embed_config, WebhookPayload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
TEI_URL = os.getenv("TEI_URL")

app = FastAPI()


@app.get("/")
async def home():
    return FileResponse("home.html")


@app.post("/webhook")
async def post_webhook(
        payload: WebhookPayload,
        task_queue: BackgroundTasks
):
    if not (
            payload.event.action == "update"
            and payload.event.scope.startswith("repo.content")
            # and payload.repo.name == chunk_config.input_dataset # any input dataset
            and payload.repo.type == "dataset"
    ):
        # no-op
        logger.info("Update detected, no action taken")
        return {"processed": False}

    if payload.repo.name == chunk_config.input_dataset:
        task_queue.add_task(chunk_dataset)
        task_queue.add_task(embed_dataset)

    return {"processed": True}


"""
CHUNKING
"""

class Chunker:
    def __init__(self, strategy, split_seq, chunk_len):
        self.split_seq = split_seq
        self.chunk_len = chunk_len
        if strategy == "spacy":
            self.split = SpacyTextSplitter().split_text
        if strategy == "sequence":
            self.split = self.seq_splitter
        if strategy == "constant":
            self.split = self.const_splitter

    def seq_splitter(self, text):
        return text.split(self.split_seq)

    def const_splitter(self, text):
        return [
            text[i * self.chunk_len:(i + 1) * self.chunk_len]
            for i in range(int(np.ceil(len(text) / self.chunk_len)))
        ]


def chunk_generator(input_dataset, chunker):
    for i in tqdm(range(len(input_dataset))):
        chunks = chunker.split(input_dataset[i][chunk_config.input_text_col])
        for chunk in chunks:
            if chunk:
                yield {chunk_config.input_text_col: chunk}


def chunk_dataset():
    logger.info("Update detected, chunking is scheduled")
    input_ds = load_dataset(chunk_config.input_dataset, split="+".join(chunk_config.input_splits))
    chunker = Chunker(
        strategy=chunk_config.strategy,
        split_seq=chunk_config.split_seq,
        chunk_len=chunk_config.chunk_len
    )

    dataset = Dataset.from_generator(
        chunk_generator,
        gen_kwargs={
            "input_dataset": input_ds,
            "chunker": chunker
        }
    )

    dataset.push_to_hub(
        chunk_config.output_dataset,
        private=chunk_config.private,
        token=HF_TOKEN
    )

    logger.info("Done chunking")

    return {"processed": True}


"""
EMBEDDING
"""

async def embed_sent(sentence, semaphore, tmp_file):
    async with semaphore:
        payload = {
            "inputs": sentence,
            "truncate": True
        }

        async with ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {HF_TOKEN}"
                }
        ) as session:
            async with session.post(TEI_URL, json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(await resp.text())
                result = await resp.json()

                tmp_file.write(
                    json.dumps({"vector": result[0], chunk_config.input_text_col: sentence}) + "\n"
                )


async def embed(input_ds, temp_file):
    semaphore = asyncio.BoundedSemaphore(embed_config.semaphore_bound)
    jobs = [
        asyncio.create_task(embed_sent(row[chunk_config.input_text_col], semaphore, temp_file))
        for row in input_ds if row[chunk_config.input_text_col].strip()
    ]
    logger.info(f"num chunks to embed: {len(jobs)}")

    tic = time.time()
    await tqdm_asyncio.gather(*jobs)
    logger.info(f"embed time: {time.time() - tic}")


def wake_up_endpoint(url):
    n_loop = 0
    while requests.get(
        url=url,
        headers={"Authorization": f"Bearer {HF_TOKEN}"}
    ).status_code != 200:
        time.sleep(2)
        n_loop += 1
        if n_loop > 10:
            raise TimeoutError("TEI endpoint is unavailable")
    logger.info("TEI endpoint is up")


def embed_dataset():
    logger.info("Update detected, embedding is scheduled")
    wake_up_endpoint(TEI_URL)
    input_ds = load_dataset(embed_config.input_dataset, split="+".join(chunk_config.input_splits))
    with tempfile.NamedTemporaryFile(mode="a", suffix=".jsonl") as temp_file:
        asyncio.run(embed(input_ds, temp_file))

        dataset = Dataset.from_json(temp_file.name)
        dataset.push_to_hub(
            embed_config.output_dataset,
            private=embed_config.private,
            token=HF_TOKEN
        )

    logger.info("Done embedding")


# For debugging

# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860)
