# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License

import json

from pydantic import BaseModel, ConfigDict


class OrganizationRelationship(BaseModel):
    """
    The model of organization and its relationship type with the author
    """
    model_config = ConfigDict(extra='forbid')
    org_name: str
    relationship_type: list[str]


class AuthorInfo(BaseModel):
    """
    The model of author and its organization relationship
    """
    model_config = ConfigDict(extra='forbid')
    author_name: str
    organization: list[OrganizationRelationship]


class Result(BaseModel):
    """
    The model of the result of the inference
    """
    model_config = ConfigDict(extra='forbid')
    author_info: list[AuthorInfo]

class AuthorInfoRequest(BaseModel):
    authors: list[str]
    disclosure: str

def build_prompt(data: AuthorInfoRequest):
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": ("You are a tool that helps extract relationship information between sponsoring entities and "
                     "authors from the disclosure statement of an article. You will be given the list of authors' "
                     "names and the disclosure statement. Extract the relationships in a JSON format.\n"
                     "Perform these steps to validate the result before printing it:\n"
                     "1. The output includes only the provided author names, exactly as they are written, "
                     "without any other names.\n"
                     "2. The eligible choices for relationship_type are:\n"
                     "['Honorarium', 'Named Professor', 'Received research materials directly', 'Patent license', "
                     "'Other/Unspecified', 'Personal fees', 'Salary support', 'Received research materials "
                     "indirectly', 'Equity', 'Expert testimony', 'Consultant', 'Board member', 'Founder of entity "
                     "or organization', 'Received travel support', 'Holds Chair', 'Fellowship', 'Scholarship', "
                     "'Collaborator', 'Received research grant funds directly', 'No Relationship', 'Speakers’ "
                     "bureau', 'Employee of', 'Received research grant funds indirectly', 'Patent', 'Award', "
                     "'Research Trial committee member', 'Supported', 'Former employee of']\n"
                     "3. There can be more than one relationship type between the author and sponsoring entity.")
            }
        ]
    }
    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f'Authors: {data.author}\nStatement: {data.coi_statement}'
            }
        ]
    }
    return [system_prompt, user_prompt]


def create_batch(dataset: list) -> list:
    """
    Create prompts for inference
    :param dataset: the dataset
    :return: the batch of prompts
    """
    prompts = []
    for line in dataset:
        data = json.loads(line.strip())
        authors = [author_data['name'] for author_data in data['authors']]
        disclosure = data['disclosure']
        data = AuthorInfoRequest(authors=authors, disclosure=disclosure)
        prompts.append(build_prompt(data))
    batch = []
    schema = Result.model_json_schema()
    for i, message in enumerate(prompts):
        data = {
            'custom_id': str(i),
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': 'ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:author-org-legal:AJ3RkyvU', #ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:author-org-legal:A5jUNqa3
                'messages': message,
                'temperature': 0,
                'max_tokens': 16384,
                'top_p': 1,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'result',
                        'schema': schema,
                        "strict": True
                    },

                }
            }
        }
        batch.append(json.dumps(data))
    return batch


def infer(client, prompt):
    """
    Infer the relationship between the author and the organization
    :param client: the OpenAI client
    :param prompt: the prompt to infer
    :return: the response from the OpenAI API (not the results)
    """
    response = client.beta.chat.completions.parse(
        model="ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:author-org-legal:AJ3RkyvU",
        messages=prompt,
        temperature=0.5,
        max_tokens=16384,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=Result
    )
    return response

def format_and_combine(dataset: list, results: list) -> None:
    """
    Format the results and combine them with the dataset
    :param dataset: the original dataset
    :param results: the results from the OpenAI inference
    :return:
    """
    for i, (x, data) in enumerate(zip(results, dataset)):
        data = json.loads(data)
        finish_reason = x['response']['body']['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            data['author_info'] = []
            continue
        result = Result(**json.loads(x['response']['body']['choices'][0]['message']['content']))
        author_info = {}
        author_id = 0
        for a_info in result['author_info']:
            author_info[author_id] = {}
            author_info[author_id]['__name'] = a_info['author_name']
            author_info[author_id]['__relationships'] = []
            for org in a_info['organization']:
                author_info[author_id]['__relationships'].append([org['org_name'], org['relationship_type']])
            author_id += 1
        data['author_info'] = author_info
        dataset[i] = json.dumps(data)