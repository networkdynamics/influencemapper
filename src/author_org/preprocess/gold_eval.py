import json

if __name__ == '__main__':
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles120.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    triples = [[(author_data['__name'], org[0][0] if type(org[0]) is list else org[0], org[1]) for _, author_data in dataset['author_info'].items() for org in author_data['__relationships']] for dataset in
               datasets ]
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles120_triplet.jsonl', 'w').writelines('\n'.join([json.dumps(triple) for triple in triples]))
    print()