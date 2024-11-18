# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License
import json


def build_prompt(authors, coi_statement):
    system_prompt = ("You are a tool that helps extract relationship information between sponsoring entities and "
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
                     "'Collaborator', 'Received research grant funds directly', 'No Relationship', 'Speakers\' "
                     "bureau', 'Employee of', 'Received research grant funds indirectly', 'Patent', 'Award', "
                     "'Research Trial committee member', 'Supported', 'Former employee of']\n"
                     "3. There can be more than one relationship type between the author and sponsoring entity.")
    user_prompt = f'Authors: {authors}\nStatement: {coi_statement}'
    return system_prompt, user_prompt

def create_prompts(dataset: list) -> list:
    """
    Create prompts for fine-tuning the model
    :param dataset: a list of jsonlines dataset
    :return: prompts containing system, user, and assistant prompts
    """
    prompts = []
    for line in dataset:
        data = json.loads(line.strip())
        authors = [author_data['__name'] for _, author_data in data['author_info'].items()]
        system_prompt, user_prompt = build_prompt(authors, data['disclosure'])
        author_info = []
        for _, author_data in data['study_info'].items():
            orgs = author_data['__relationships']
            orgs_rel = {o[0]: [] for o in orgs}
            for org in orgs:
                org_name = org[0]
                relationship_type = org[1]
                orgs_rel[org_name].append(relationship_type)
            author_info.append([(org, rels) for org, rels in orgs_rel.items()])
        assistant_prompt = json.dumps({
            'author_info': [
                {'author_name': authors, 'organization': [{'org_name': org[0], 'relationship_type': org[1]}
                                                          for org in author_info]}]
        })
        prompts.append((system_prompt, user_prompt, assistant_prompt))
    return prompts