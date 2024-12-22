# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License
import json
from typing import Tuple


def build_prompt(disclosure: str) -> Tuple[str, str]:
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
    user_prompt = f'Statement: {disclosure}'
    return system_prompt, user_prompt

def create_prompts(dataset: list) -> list:
    """
    Create prompts for fine-tuning the model
    :param dataset: a list of jsonlines dataset
    :return: prompts containing system, user, and assistant prompts
    """
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
    prompts = []
    for line in dataset:
        data = json.loads(line.strip())
        system_prompt, user_prompt = build_prompt(data['disclosure'])
        study_info = []
        for org, rels in data['study_info'].items():
            org_rel = {
                'org_name': org,
                'relationships': [{'relationship_type': reversed_rel_map[rel],
                                   'relationship_indication': 'yes' if 'yes' in answer else answer} for rel, answer in
                                  rels.items() if answer != 'unknown']
            }
            study_info.append(org_rel)
        assistant_prompt = json.dumps({'study_info': study_info})
        prompts.append((system_prompt, user_prompt, assistant_prompt))
    return prompts



