import json

from openai import OpenAI
from tqdm import tqdm


def build_prompt(author, coi_statement):
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
    user_prompt = f'Authors: {author}\nStatement: {coi_statement}'
    return system_prompt, user_prompt