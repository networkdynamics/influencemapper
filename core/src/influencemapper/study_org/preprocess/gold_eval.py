import json

if __name__ == '__main__':
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/valid.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    result = []
    for dataset in datasets:
        pairs = []
        for org, rels in dataset['study_info'].items():
            for rel, answer in rels.items():
                answer = 'yes' if 'yes' in answer else answer
                pairs.append((org, rel, answer))
        result.append(json.dumps(pairs))
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/study_org/valid_triples.jsonl', 'w').writelines('\n'.join(result))
    print()