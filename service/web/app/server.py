import asyncio
import json
from contextlib import asynccontextmanager

import aioredis
import redis
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pandas import DataFrame
from pydantic import BaseModel
import pandas as pd
import io

redis_client = None
pubsub = None

async def get_redis_client():
    return await aioredis.from_url("redis://redis:6379")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, pubsub
    redis_client = await get_redis_client()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('author_result')
    yield
    redis_client.close()
    await redis_client.wait_closed()

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")

class AuthorInfoRequest(BaseModel):
    authors: list[str]
    disclosure: str

class StudyInfoRequest(BaseModel):
    disclosure: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def publish_infos(df: DataFrame, channel: str, transform_func):
    infos = df.groupby("Article title").apply(transform_func).to_dict()
    for key, value in infos.items():
        redis_client.publish(channel, json.dumps(value.dict()))
    # async for message in pubsub_author.listen():
    #     if message["type"] == "message":
    #         result = json.loads(message["data"])
    #         print("Received result:", result)
    #         return {"result": result}

async def publish_author_infos(df: DataFrame):
    await publish_infos(
        df,
        'author_channel',
        lambda group: AuthorInfoRequest(
            authors=group["Author Name"].tolist(),
            disclosure=group["Disclosure Statement"].iloc[0]
        )
    )

async def publish_study_infos(df: DataFrame):
    await publish_infos(
        df,
        'study_channel',
        lambda group: StudyInfoRequest(
            disclosure=group["Disclosure Statement"].iloc[0]
        )
    )

@app.post('/upload')
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "File is not a CSV"}
    try:
        # Read the uploaded CSV
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        asyncio.create_task(publish_infos(df, 'author_channel', publish_author_infos))
        asyncio.create_task(publish_infos(df, 'study_channel', publish_study_infos))
        print("Waiting for results...")
        return "Data uploaded successfully"
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
