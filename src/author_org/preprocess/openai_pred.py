import json

if __name__ == '__main__':
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/valid_openai_4omini_ft.jsonl').readlines()
    data = [json.loads(line) for line in data]
    datasets = []
    for i, x in enumerate(data):
        finish_reason = x['response']['body']['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            datasets.append({'author_info': []})
            continue
        datasets.append(json.loads(x['response']['body']['choices'][0]['message']['content']))
    # datasets = [json.loads(json.loads(line)['response']['body']['choices'][0]['message']['content']) for line in data]
    triples = [[(item['author_name'], org['org_name'], org['relationship_type']) for item in dataset['author_info'] for org in item['organization']] for dataset in
               datasets]
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/valid_openai_4omini_ft_triples.jsonl', 'w').writelines('\n'.join([json.dumps(triple) for triple in triples]))
    print()