import asyncio
import json
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

import server
import listener
import pandas as pd


def mock_infer(client, prompt):
    mock_result = {
        'response': {
            'body': {
                'choices': [
                    {
                        'finish_reason': 'stop',
                        'message': {
                            'content': json.dumps('mocked response')
                        }
                    }
                ]
            }
        }
    }
    return mock_result

def get_client():
    with open('secret_key') as f:
        secret_key = f.read().strip()
        client = OpenAI(api_key=secret_key)
    return client

@pytest.mark.asyncio
async def test_infer_study(redisdb, monkeypatch):
    client = get_client()
    df = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/sample_data/Biomedical_Papers.csv')
    monkeypatch.setattr(server, 'redis_client', redisdb)
    pubsub = redisdb.pubsub()
    pubsub.subscribe(*['study_channel', 'study_result'])
    await asyncio.sleep(1)
    pubsub.get_message()
    pubsub.get_message()
    await server.publish_infos(df, 'study_channel')
    await asyncio.sleep(1)
    message = pubsub.get_message()
    data = json.loads(message['data'].decode('utf-8'))
    monkeypatch.setattr(listener, 'infer_study', mock_infer)
    listener.process_message(redisdb, data, client, 'study_channel')
    await asyncio.sleep(1)
    message = pubsub.get_message()
    assert message['data'].decode('utf-8') == json.dumps('mocked response')

@pytest.mark.asyncio
async def test_infer_author(redisdb, monkeypatch):
    client = get_client()
    df = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/sample_data/Biomedical_Papers.csv')
    monkeypatch.setattr(server, 'redis_client', redisdb)
    pubsub = redisdb.pubsub()
    pubsub.subscribe(*['author_channel', 'author_result'])
    await asyncio.sleep(1)
    pubsub.get_message()
    pubsub.get_message()
    await server.publish_infos(df, 'author_channel')
    await asyncio.sleep(1)
    message = pubsub.get_message()
    data = json.loads(message['data'].decode('utf-8'))
    monkeypatch.setattr(listener, 'infer_author', mock_infer)
    listener.process_message(redisdb, data, client, 'author_channel')
    await asyncio.sleep(1)
    message = pubsub.get_message()
    assert message['data'].decode('utf-8') == json.dumps('mocked response')
