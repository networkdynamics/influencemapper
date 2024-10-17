import json
import logging

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class Organization(BaseModel):
    model_config = ConfigDict(extra='forbid')
    org_name: str
    relationship_type: list[str]


class AuthorInfo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    author_name: str
    organization: list[Organization]


class Result(BaseModel):
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


def create_batch(ids, messages):
    batch = []
    schema = Result.model_json_schema()
    for i, message in zip(ids, messages):
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


def main():
    api_key = open('/Users/blodstone/Research/influencemapper/InfluenceMapper/secret_key').read().strip()
    client = OpenAI(api_key=api_key)
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles134.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    authors = [[author['name'] for author in dataset['authors']] for dataset in datasets ]
    coi_statements = [dataset['coi_statements'] if type(dataset['coi_statements']) is list else dataset['coi_statements'] for dataset in datasets]
    messages = [build_prompt(author, coi_statement) for author, coi_statement in zip(authors, coi_statements) if coi_statement != '']
    batch = create_batch([str(i) for i in range(len(messages))], messages)
    batch_name = 'batch_134.jsonl'
    open(batch_name, "w").write('\n'.join(batch))
    batch_input_file = client.files.create(
        file=open(batch_name, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": 'Batch Author Org'
        }
    )
    # responses = [infer(client, message) for message in messages]
    # results = [json.loads(response.choices[0].message.content) for response in responses]
    print()


if __name__ == '__main__':
    logging.info('Starting program')
    main()
