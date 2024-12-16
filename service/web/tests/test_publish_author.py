import asyncio
import json

import pytest

import service.web.app.server as server
import pandas as pd

@pytest.mark.asyncio
async def test_publish(redisdb, monkeypatch):
    df = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/sample_data/Biomedical_Papers.csv')
    monkeypatch.setattr(server, 'redis_client', redisdb)
    pubsub = redisdb.pubsub()
    pubsub.subscribe('author_channel')
    pubsub.get_message()
    monkeypatch.setattr(server, 'pubsub', pubsub)
    await server.publish_author_infos(df)
    await asyncio.sleep(1)
    message = pubsub.get_message()
    exp_data = {"authors": ["Dr. John Smith", "Dr. Emily Johnson"], "disclosure": "The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported."}
    assert json.loads(message['data']) == exp_data