# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License

import json

from pydantic import BaseModel, ConfigDict


class Relationship(BaseModel):
    model_config = ConfigDict(extra='forbid')
    relationship_type: str
    relationship_indication: str


class StudyInfo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    org_name: str
    relationships: list[Relationship]


class Result(BaseModel):
    model_config = ConfigDict(extra='forbid')
    study_info: list[StudyInfo]


class StudyInfoRequest(BaseModel):
    disclosure: str


def build_prompt(data: StudyInfoRequest):
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a tool that helps extract relationship information between sponsoring entities and "
                        "the study from the disclosure statement of an article. You will be given the disclosure "
                        "statement. Extract the relationships in a JSON format.\n "
                        "Perform these steps to validate the result before printing it:\n"
                        "1. The eligible choices for relationship_type are: "
                        "['Perform analysis', 'Collect data', 'Coordinate the study', 'Design the study', "
                        "'Fund the study', 'Participate in the study', 'Review the study', 'Supply the study',"
                        "'Supply data to the study', 'Support the study', 'Write the study', 'Other']\n"
                        "2. The eligible choices for relationship_indication are: "
                        "['Yes', 'No']\n"
                        "3. The output must only use the listed relationship_type exactly as they are written, "
                        "without any other names.\n"
                        "4. There can be more than one relationship_type per organization."

            }
        ]
    }
    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f'{data.disclosure}'
            }
        ]
    }
    return [system_prompt, user_prompt]


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
            data['study_info'] = []
            continue
        result = Result(**json.loads(x['response']['body']['choices'][0]['message']['content']))
        study_info = {}
        for s_info in result['study_info']:
            relationships = {}
            for rel in s_info['relationships']:
                relationships[rel['relationship_type']] = rel['relationship_indication']
            study_info[s_info['org_name']] = relationships
        data['study_info'] = study_info
        dataset[i] = json.dumps(data)


def create_batch(dataset: list):
    prompts = []
    for line in dataset:
        data = json.loads(line.strip())
        disclosure = data['disclosure']
        data = StudyInfoRequest(disclosure=disclosure)
        prompts.append(build_prompt(data))
    batch = []
    schema = Result.model_json_schema()
    for i, message in enumerate(prompts):
        data = {
            'custom_id': str(i),
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': 'ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:study-org:A0zjJe9i',
                'messages': message,
                'temperature': 0.5,
                'max_tokens': 16384,
                'top_p': 0.9,
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
        model="ft:gpt-4o-mini-2024-07-18:network-dynamics-lab:study-org:A0zjJe9i",
        messages=message,
        temperature=0.5,
        max_tokens=16384,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=Result
    )
    return response
