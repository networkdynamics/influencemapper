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