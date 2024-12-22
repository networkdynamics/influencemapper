import json

import pytest
from openai import OpenAI

from author_org.infer import infer as infer_author, build_prompt as build_prompt_author, AuthorInfoRequest
from study_org.infer import infer as infer_study, build_prompt as build_prompt_study, StudyInfoRequest


@pytest.fixture
def client():
    with open('tests/secret_key') as f:
        secret_key = f.read().strip()
        client = OpenAI(api_key=secret_key)
    return client

@pytest.fixture
def author_data():
    authors = ['Dr. John Smith', 'Dr. Jane Doe']
    return authors

@pytest.fixture
def disclosure_data():
    disclosure = 'The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported.'
    return disclosure

def test_infer_author(client, author_data, disclosure_data):
    result = infer_author(client=client, prompt=build_prompt_author(AuthorInfoRequest(authors=author_data, disclosure=disclosure_data)))
    assert result.choices[0].finish_reason == 'stop'
    assert json.loads(result.choices[0].message.content) != {}

def test_infer_study(client, author_data, disclosure_data):
    result = infer_study(client=client, prompt=build_prompt_study(StudyInfoRequest(disclosure=disclosure_data)))
    assert result.choices[0].finish_reason == 'stop'
    assert json.loads(result.choices[0].message.content) != {}