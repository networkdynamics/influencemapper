import json

if __name__ == '__main__':
    predict = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/batch_R2VQRvIBHAElUFxcP1bpqpoY_output.jsonl').readlines()
    for data in predict:
        data = json.loads(data)
        response = data['response']
        result = json.loads(response['body']['choices'][0]['message']['content'])
        print()