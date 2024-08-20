import json

from openai import OpenAI
from tqdm import tqdm


def build_prompt(author, coi_statement):
    system_prompt = """You are a tool that helps extract relationship information between sponsoring entities and 
                    authors from the disclosure statement of an article. You will be given the list of authors' 
                    names and the disclosure statement. Extract the relationships in a JSON format.\n 
                    Perform these steps to validate the result before printing it:\n
                    1. The output includes only the provided author names, exactly as they are written, 
                    without any other names.\n
                    2. The eligible choices for relationship_type are:
                    ['Honorarium', 'Named Professor', 'Received research materials directly', 'Patent license', 
                    'Other/Unspecified', 'Personal fees', 'Salary support', 'Received research materials 
                    indirectly', 'Equity', 'Expert testimony', 'Consultant', 'Board member', 'Founder of entity 
                    or organization', 'Received travel support', 'Holds Chair', 'Fellowship', 'Scholarship', 
                    'Collaborator', 'Received research grant funds directly', 'No Relationship', 'Speakers’ 
                    bureau', 'Employee of', 'Received research grant funds indirectly', 'Patent', 'Award', 
                    'Research Trial committee member', 'Supported', 'Former employee of']"""
    user_prompt = f'Authors: {author}\nStatement: {coi_statement}'
    return system_prompt, user_prompt


if __name__ == '__main__':
    api_key = open('/Users/blodstone/Research/influencemapper/InfluenceMapper/secret_key').read().strip()
    client = OpenAI(api_key=api_key)
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/train_repaired_resolved.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    authors = [[author_data['__name'] for _, author_data in dataset['author_info'].items()] for dataset in
               datasets]
    relationships = [
        [[(org[0][0] if type(org[0]) is list else org[0], org[1]) for org in author_data['__relationships']] for
         _, author_data in dataset['author_info'].items()] for dataset in datasets]
    coi_statements = [' '.join(dataset['coi_statements']) for dataset in datasets]
    outputs = []
    valid = False
    for author, relationship, coi_statement in tqdm(zip(authors, relationships, coi_statements)):
        if valid or sum([sum([1 for org in r]) for r in relationship]) != 0:
            system_prompt, user_prompt = build_prompt(author, coi_statement)
            assistant_prompt = {
                'author_info': [
                    {'author_name': a, 'organization': [{'org_name': org[0], 'relationship_type': org[1]} for org in r]} for
                    a, r in zip(author, relationship)]
            }
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
    open("train_finetune.jsonl", "w").write('\n'.join(outputs))
    # client.files.create(
    #     file=open("finetune.jsonl", "rb"),
    #     purpose="fine-tune"
    # )
    # client.fine_tuning.jobs.create(
    #     training_file="finetune.jsonl",
    #     model="gpt-4o-mini"
    # )
