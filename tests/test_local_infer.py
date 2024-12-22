import json

import pytest

from author_org.infer import infer as infer_author, build_prompt as build_prompt_author, AuthorInfoRequest, create_batch as create_batch_author
from study_org.infer import infer as infer_study, create_batch as create_batch_study


@pytest.fixture
def my_client(mocker):
    mock_client = mocker.MagicMock()
    mock_client.beta.chat.completions.parse.return_value = {
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
    return mock_client

@pytest.fixture
def author_data():
    authors = ['Dr. John Smith', 'Dr. Jane Doe']
    return authors

@pytest.fixture
def disclosure_data():
    disclosure = 'The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported.'
    return disclosure

@pytest.fixture
def dataset():
    author_1 = {
        'authors': [{'name': 'Dr. John Smith'}, {'name': 'Dr. Jane Doe'}],
        'disclosure': 'The author declares no conflict of interest. This research was funded by the National Cancer Institute. No financial or non-financial competing interests are reported.'
    }
    author_2 = {
        'authors': [{'name': 'Dr. Roe White'}, {'name': 'Dr. Betty Black'}],
        'disclosure': 'The authors declare the following potential conflicts of interest: Dr. Roe White has received research funding from Pfizer Inc. and consulting fees from Merck & Co., Inc. Dr. Betty Black is a shareholder in Johnson & Johnson and has received travel support from GlaxoSmithKline.'
    }
    dataset = [
        json.dumps(author_1),
        json.dumps(author_2)
    ]
    return dataset


def test_build_prompt(author_data, disclosure_data):
    prompts = build_prompt_author(AuthorInfoRequest(authors=author_data, disclosure=disclosure_data))
    assert len(prompts) == 2
    system_prompt = prompts[0]
    user_prompt = prompts[1]
    assert 'content' in system_prompt
    assert user_prompt['content'][0]['text'] == f'Authors: {author_data}\nStatement: {disclosure_data}'


def test_infer_author(my_client):
    response = infer_author(client=my_client, prompt='test')
    assert response == {'response': {'body': {'choices': [{'finish_reason': 'stop', 'message': {'content': '"mocked response"'}}]}}}

def test_infer_study(my_client):
    response = infer_study(client=my_client, prompt='test')
    assert response == {'response': {'body': {'choices': [{'finish_reason': 'stop', 'message': {'content': '"mocked response"'}}]}}}

def test_create_batch_author(dataset):
    batch = create_batch_author(dataset)
    assert len(batch) == len(dataset)

def test_create_batch_study(dataset):
    batch = create_batch_study(dataset)
    assert len(batch) == len(dataset)