import asyncio
import json
from pathlib import Path

import pytest

import server
import pandas as pd

@pytest.fixture
def test_df():
    resource_path = Path(__file__).parent / 'resources' / 'data.csv'
    return pd.read_csv(resource_path)

@pytest.mark.asyncio
async def test_publish_author(redisdb, monkeypatch, test_df):
    channel_name = 'author_channel'
    monkeypatch.setattr(server, 'redis_client', redisdb)
    pubsub = redisdb.pubsub()
    pubsub.subscribe(channel_name)
    await asyncio.sleep(1)
    pubsub.get_message()
    await server.publish_infos(test_df, channel_name)
    await asyncio.sleep(1)
    message = pubsub.get_message()
    exp_data = {"authors": ["Dr. John Smith", "Dr. Emily Johnson"], "disclosure": "The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported."}
    assert json.loads(message['data'].decode('utf-8')) == exp_data

@pytest.mark.asyncio
async def test_publish_study(redisdb, monkeypatch):
    channel_name = 'study_channel'
    monkeypatch.setattr(server, 'redis_client', redisdb)
    pubsub = redisdb.pubsub()
    pubsub.subscribe(channel_name)
    await asyncio.sleep(1)
    pubsub.get_message()
    await server.publish_infos(test_df, channel_name)
    await asyncio.sleep(1)
    message = pubsub.get_message()
    exp_data = {"disclosure": "The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported."}
    assert json.loads(message['data'].decode('utf-8')) == exp_data