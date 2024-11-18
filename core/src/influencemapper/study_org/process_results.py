import json


class StudyOrgResult:

    def __init__(self):
        self.dataset = []

    def process_result_batch(self, filename: str):
        for line in open(filename):
            data = json.loads(line)
            finish_reason = data['response']['body']['choices'][0]['finish_reason']
            if finish_reason != 'stop':
                self.dataset.append({'study_info': []})
                continue
            self.dataset.append(json.loads(data['response']['body']['choices'][0]['message']['content']))
        return self.dataset
