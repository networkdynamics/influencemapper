import json
import logging
import os
from argparse import ArgumentParser
from typing import Tuple

import tiktoken
from openai import OpenAI
from tqdm import tqdm

from influencemapper.author_org.infer import create_batch as author_org_create_batch
from influencemapper.study_org.infer import create_batch as study_org_create_batch
from influencemapper.study_org.fine_tune import create_prompts as study_org_create_prompts
from influencemapper.author_org.fine_tune import create_prompts as author_org_create_prompts


def create_fine_tune_batch(data_path: str, purpose:str, model_name: str, threshold: int) -> Tuple[list, int]:
    """
    Create a fine-tune file for the OpenAI batch fine-tuning API
    :param data_path: the path to the training data
    :param purpose: whether for study-org or author-org model
    :param model_name: the OpenAI model name, default is gpt-4o-mini.
    :param threshold: the threshold for the number of tokens in a batch. Above this threshold, the line is skipped
    The model name is used to count the number of tokens needed to run fine-tuning
    """
    data = open(data_path).readlines()
    if purpose == 'study_org':
        prompts = study_org_create_prompts(data)
    elif purpose == 'author_org':
        prompts = author_org_create_prompts(data)
    else:
        raise ValueError(f"Invalid purpose: {purpose}")
    outputs = []
    total_tokens = 0
    encoding = tiktoken.encoding_for_model(model_name)
    for system_prompt, user_prompt, assistant_prompt in tqdm(prompts):
        tokens = len(encoding.encode(json.dumps(assistant_prompt['study_info'])))
        if tokens > threshold:
            continue
        total_tokens += tokens
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
    return outputs, total_tokens

def generate_openai_files(train_data: str, valid_data:str, purpose: str, model_name: str, threshold: int) -> None:
    """
    Generate the files for OpenAI fine-tuning
    :param train_data: the path to the training data
    :param valid_data: the path to the validation data
    :param purpose: whether for study-org or author-org model
    :param model_name: the OpenAI model name, default is gpt-4o-mini.
    :param threshold: the threshold for the number of tokens in a batch. Above this threshold, the line is skipped
    """
    logging.info("Generating OpenAI fine-tuning files")
    logging.info(f"Training data: {train_data}")
    train, train_tokens = create_fine_tune_batch(train_data, purpose, model_name, threshold)
    train_dir = os.path.dirname(train_data)
    open(os.path.join(train_dir, 'train_finetune.jsonl'), "w").write('\n'.join(train))
    logging.info(f"Total tokens for training data: {train_tokens}")
    logging.info(f"Saving training data as: {os.path.join(train_dir, 'train_finetune.jsonl')}")
    logging.info(f"Validation data: {valid_data}")
    valid, valid_tokens = create_fine_tune_batch(valid_data, purpose, model_name, threshold)
    valid_dir = os.path.dirname(valid_data)
    open(os.path.join(valid_dir, 'valid_finetune.jsonl'), "w").write('\n'.join(valid))
    logging.info("Upload the files to OpenAI dashboard to fine-tune the model: https://platform.openai.com/finetune/")
    logging.info(f"Total tokens for validation data: {valid_tokens}")
    logging.info(f"Saving validation data as: {os.path.join(valid_dir, 'valid_finetune.jsonl')}")


def submit_batch_to_openai(api_key: str, dataset_path: str, purpose: str) -> None:
    """
    Submit a batch to OpenAI
    :param dataset_path: dataset for the batch
    :param purpose: whether for study-org or author-org model
    :return:
    """
    logging.info("Submitting batch to OpenAI")
    client = OpenAI(api_key=api_key)
    data = open(dataset_path).readlines()
    if purpose == 'study_org':
        batch = study_org_create_batch(data)
    elif purpose == 'author_org':
        batch = author_org_create_batch(data)
    else:
        raise ValueError(f"Invalid purpose: {purpose}")
    batch_name = f'batch_{purpose}.jsonl'
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
            "description": f'Batch for {purpose}'
        }
    )
    logging.info(f"Batch Info: {batch_info}")

if __name__ == "__main__":
    parser = ArgumentParser(prog='Influence Mapper CLI commands',
                            description='Run InfluenceMapper on a preset dataset')
    subparsers = parser.add_subparsers(dest='command', required=True, help='CLI commands')

    fine_tune = subparsers.add_parser('fine_tune', help='Fine-tune OpenAI GPT model using dataset')
    infer = subparsers.add_parser('infer', help='Infer COI statements using OpenAI GPT model')
    evaluate = subparsers.add_parser('evaluate', help='Evaluate the performance of the model')

    fine_tune.add_argument('-train_data', type=str, required=True, help='Path to training data')
    fine_tune.add_argument('-valid_data', type=str, required=True, help='Path to validation data')
    fine_tune.add_argument('-model_name', type=str, default='gpt-4o-mini', help='OpenAI model name')
    fine_tune.add_argument('-threshold', type=int, default=1500,
                           help='Threshold for the number of tokens in a batch')
    fine_tune.add_argument('-API_KEY', type=str, required=True, help='OpenAI API key')

    ft_parser = fine_tune.add_subparsers(dest='purpose', required=True, help='Fine-tune model')
    ft_study_org = ft_parser.add_parser('study_org', help='Fine-tune study-org model')
    ft_author_org = ft_parser.add_parser('author_org', help='Fine-tune author-org model')

    infer_parser = infer.add_subparsers(dest='purpose', required=True, help='Infer COI statements')
    infer_study_org = infer_parser.add_parser('study_org', help='Infer COI statements using study-org model')
    infer_author_org = infer_parser.add_parser('author_org', help='Infer COI statements using author-org model')

    evaluate_parser = evaluate.add_subparsers(dest='purpose', required=True, help='Evaluate model')
    evaluate_study_org = evaluate_parser.add_parser('study_org', help='Evaluate study-org model')
    evaluate_author_org = evaluate_parser.add_parser('author_org', help='Evaluate author-org model')

    args = parser.parse_args()

    if args.command == 'fine_tune':
        generate_openai_files(args.train_data, args.valid_data, args.purpose, args.model_name, args.threshold)
    elif args.command == 'infer':
        submit_batch_to_openai(args.API_KEY, args.dataset_path, args.purpose)
    elif args.command == 'evaluate':
        if args.purpose == 'study_entity':
            print('Evaluating study-entity model')
        elif args.purpose == 'author_entity':
            print('Evaluating author-entity model')


