import json

import tiktoken
from openai import OpenAI
from tqdm import tqdm


def build_prompt(coi_statement):
    system_prompt = ("You are a tool that helps extract relationship information between sponsoring entities and "
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
                     "4. There can be more than one relationship_type per organization.")
    user_prompt = f'Statement: {coi_statement}'
    return system_prompt, user_prompt



if __name__ == '__main__':
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    api_key = open('/Users/blodstone/Research/influencemapper/InfluenceMapper/secret_key').read().strip()
    client = OpenAI(api_key=api_key)
    data_path = '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/train.jsonl'
    data = open(data_path).readlines()
    datasets = [json.loads(line) for line in data]
    coi_statements = [' '.join(dataset['coi_statements']) for dataset in datasets]
    reversed_rel_map = {
        'analyzed': 'Perform analysis',
        'collected_data': 'Collect data',
        'coordinated': 'Coordinate the study',
        'designed': 'Design the study',
        'funded': 'Fund the study',
        'participated_in': 'Participate in the study',
        'reviewed': 'Review the study',
        'supplied': 'Supply the study',
        'supplied_data': 'Supply data to the study',
        'supported': 'Support the study',
        'wrote': 'Write the study',
        'other': 'Other'
    }
    relationships = []
    for dataset in datasets:
        rows = []
        for org, rels in dataset['study_info'].items():
            org_rel = {
                'org_name': org,
                'relationships': [{'relationship_type': reversed_rel_map[rel], 'relationship_indication': 'yes' if 'yes' in answer else answer} for rel, answer in rels.items() if answer != 'unknown']
            }
            rows.append(org_rel)
        relationships.append(rows)
    outputs = []
    valid = True if 'valid' in data_path else False
    total_tokens = 0
    max_tokens = 0
    for relationship, coi_statement in tqdm(zip(relationships, coi_statements)):
        system_prompt, user_prompt = build_prompt(coi_statement)
        assistant_prompt = {
            'study_info': relationship
        }
        tokens = len(encoding.encode(json.dumps(assistant_prompt['study_info'])))
        if tokens > 1500:
             continue
        total_tokens += tokens
        max_tokens = max(max_tokens, tokens)
        output = {
            'messages': [{
                'role': 'system',
                'content': system_prompt
            }, {
                'role': 'user',
                'content': user_prompt
            }, {
                'role': 'assistant',
                'content': json.dumps(assistant_prompt)
            }]
        }
        outputs.append(json.dumps(output))
    open("/Users/blodstone/Research/influencemapper/InfluenceMapper/data/study_org/train_finetune.jsonl", "w").write('\n'.join(outputs))
    print(len(outputs))
    print(total_tokens/len(outputs))
    print(max_tokens)

    # client.files.create(
    #     file=open("finetune.jsonl", "rb"),
    #     purpose="fine-tune"
    # )
    # client.fine_tuning.jobs.create(
    #     training_file="finetune.jsonl",
    #     model="gpt-4o-mini"
    # )
