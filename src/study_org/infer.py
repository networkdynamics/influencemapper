import json
import logging

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


class Relationship(BaseModel):
    model_config = ConfigDict(extra='forbid')
    relationship_type: str
    relationship_indication: str


class StudyInfo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    org_name: str
    relationship_type: list[Relationship]


class Result(BaseModel):
    model_config = ConfigDict(extra='forbid')
    study_info: list[StudyInfo]


def build_prompt(coi_statement):
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
                "text": f'{coi_statement}'
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
                'model': 'gpt-4o-mini',
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
        model="gpt-4o-mini",
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
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/valid.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    coi_statements = [' '.join(dataset['coi_statements']) for dataset in datasets]
    messages = [build_prompt(coi_statement) for coi_statement in coi_statements]
    batch = create_batch([str(i) for i in range(len(messages))], messages)
    open("study_org_batch.jsonl", "w").write('\n'.join(batch))
    batch_input_file = client.files.create(
        file=open("study_org_batch.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "eval job"
        }
    )
    # responses = [infer(client, message) for message in messages]
    # results = [json.loads(response.choices[0].message.content) for response in responses]
    print()


if __name__ == '__main__':
    logging.info('Starting program')
    main()
