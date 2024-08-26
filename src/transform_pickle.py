import pickle
import pandas as pd
import json
if __name__ == '__main__':
    df = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/test.csv')
    dataset = []
    for index, row in df.iterrows():
        dataset.append(json.dumps(row.to_dict()))
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/test.jsonl', 'w').writelines('\n'.join(dataset))