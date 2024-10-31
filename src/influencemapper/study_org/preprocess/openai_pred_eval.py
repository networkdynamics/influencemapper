import json
rel_map = {
        'Perform analysis': 'analyzed',
        'Collect data': 'collected_data',
        'Coordinate the study': 'coordinated',
        'Design the study': 'designed',
        'Fund the study': 'funded',
        'Participate in the study': 'participated_in',
        'Review the study': 'reviewed',
        'Supply the study': 'supplied',
        'Supply data to the study': 'supplied_data',
        'Support the study': 'supported',
        'Write the study': 'wrote',
        'Other': 'other'
    }
if __name__ == '__main__':
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/temp/batch_BLeOswE4N0PkcrkYiCM27qLN_output.jsonl').readlines()
    data = [json.loads(line) for line in data]
    datasets = []
    for i, x in enumerate(data):
        finish_reason = x['response']['body']['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            datasets.append({'study_info': []})
            continue
        datasets.append(json.loads(x['response']['body']['choices'][0]['message']['content']))
    # datasets = [json.loads(json.loads(line)['response']['body']['choices'][0]['message']['content']) for line in data]
    result = []
    for dataset in datasets:
        pairs = []
        for item in dataset['study_info']:
            used_rels = set()
            for rel in item['relationships']:
                mapped_rel = rel_map[rel['relationship_type']] if rel['relationship_type'] in rel_map else 'other'
                pairs.append((item['org_name'], mapped_rel, rel['relationship_indication'].lower()))
                used_rels.add(mapped_rel)
            for rel in rel_map.values():
                if rel not in used_rels:
                    pairs.append((item['org_name'], rel, 'unknown'))
        result.append(json.dumps(pairs))
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/study_org/valid_openai_4omini_10_triples.jsonl', 'w').writelines('\n'.join(result))
    print()