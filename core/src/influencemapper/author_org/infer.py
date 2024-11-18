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


def build_prompt(author, coi_statement):
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
                "text": f'Authors: {author}\nStatement: {coi_statement}'
            }
        ]
    }
    return [system_prompt, user_prompt]


def create_batch(dataset: list):
    prompts = []
    for line in dataset:
        data = json.loads(line.strip())
        authors = [author_data['name'] for author_data in data['authors']]
        disclosure = data['disclosure']
        prompts.append(build_prompt(authors, disclosure))
    batch = []
    schema = Result.model_json_schema()
    for i, message in enumerate(prompts):
        data = {
            'custom_id': i,
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


def infer(client, message):
    response = client.beta.chat.completions.parse(
        model="ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:author-org-legal:AJ3RkyvU",
        messages=message,
        temperature=0.5,
        max_tokens=16384,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=Result
    )
    return response

def reformat_results(results):
    pass