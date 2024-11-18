# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License

import json

from influencemapper.util import RelationshipCollapsed, get_unique_map


# class Evaluate:
#
#     def __init__(self, dataset_path):
#         if not dataset_path:
#             raise ValueError("Dataset path is required")
#         self.dataset = [json.loads(line) for line in open(dataset_path)]
#         self.__process_data()
#
#     @classmethod
#     def calculate_recall_precision(cls, gold_tuples, prediction_tuples):
#         recall = len(set(gold_tuples) & set(prediction_tuples)) / len(set(gold_tuples))
#         precision = len(set(gold_tuples) & set(prediction_tuples)) / len(set(prediction_tuples))
#         return recall, precision
#
#     @classmethod
#     def calculate_component(cls, gold_tuples, prediction_tuples):
#         tp = len(set(gold_tuples) & set(prediction_tuples))
#         fp = len(set(prediction_tuples) - set(gold_tuples))
#         fn = len(set(gold_tuples) - set(prediction_tuples))
#         return tp, fp, fn
#
#     def normalize_data(self, data):
#         pass
#
#
#     @staticmethod
#     def expand_entries(data):
#         result = []
#
#         authors = data.get("author_field", [])
#         companies = data.get("company_field", [])
#         relationships = data.get("relationship_type_field", [])
#
#         for author in authors:
#             for company in companies:
#                 for relationship in relationships:
#                     result.append({
#                         "author_field": author,
#                         "company_field": company,
#                         "relationship_type_field": relationship
#                     })
#
#         return result
#
#     # def correct_entries(self, data):
#     #     new_data = []
#     #     for item2 in data:
#     #         if 'author_field' not in item2:
#     #             item2['author_field'] = []
#     #         if 'company_field' not in item2:
#     #             item2['company_field'] = []
#     #         if 'relationship_type_field' not in item2:
#     #             item2['relationship_type_field'] = []
#     #         if type(item2['author_field']) == str:
#     #             item2['author_field'] = [item2['author_field']]
#     #         if type(item2['company_field']) == str:
#     #             item2['company_field'] = [item2['company_field']]
#     #         if type(item2['relationship_type_field']) == str:
#     #             item2['relationship_type_field'] = [item2['relationship_type_field']]
#     #         new_data.extend(self.expand_entries(item2))
#     #     return new_data
#
#     def evaluate(self):
#         total_recall = []
#         total_precision = []
#         total_tp = 0
#         total_fp = 0
#         total_fn = 0
#         for gold_tuple, prediction_tuple in zip(self.gold_tuples, self.prediction_tuples):
#             if len(gold_tuple) == 0:
#                 continue
#             if len(prediction_tuple) == 0:
#                 total_recall.append(0)
#                 total_precision.append(0)
#             else:
#                 recall, precision = self.__calculate_recall_precision(gold_tuple, prediction_tuple)
#                 tp, fp, fn = self.__calculate_component(gold_tuple.split('\t'), prediction_tuple.split('\t'))
#                 total_tp += tp
#                 total_fp += fp
#                 total_fn += fn
#                 total_recall.append(recall)
#                 total_precision.append(precision)
#         return {
#             "micro_recall": total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0,
#             "micro_precision": total_tp / (total_fp + total_tp) if (total_fp + total_tp) != 0 else 0,
#             "macro_recall": sum(total_recall) / len(total_recall),
#             "macro_precision": sum(total_precision) / len(total_precision)
#         }



def evaluate(gold_triples, predict_triples, mode=3):
    total_recall = []
    total_precision = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    rc = RelationshipCollapsed()
    for gold, predict in zip(gold_triples, predict_triples):
        org_entry = {}
        orgs = set([entry[1] for entry in gold])
        orgs.update(set([entry[1] for entry in predict]))
        unique_map = get_unique_map(orgs)
        for idx, org in enumerate(orgs):
            org_entry[org] = idx
        if mode == 1:
            gold_tuples = [(entry[0]) for entry
                           in
                           gold]
            prediction_tuples = [(entry[0]) for
                                 entry
                                 in predict if (entry[0]) in gold_tuples]
        elif mode == 2:
            gold_tuples = [(entry[0], unique_map[entry[1]]) for entry in
                           gold]
            prediction_tuples = [(entry[0], unique_map[entry[1]]) for entry
                                 in predict]
        elif mode == 2.5:
            gold_tuples = [(unique_map[entry[1]]) for entry in
                           gold]
            prediction_tuples = [(unique_map[entry[1]]) for entry
                                 in predict]
        elif mode == 3.5:
            unique_author = [entry[0] for entry in gold]
            try:
                gold_tuples = [(entry[0], unique_map[entry[1]], rc.collapse_relationship(entry[2])) for entry in gold]
            except:
                gold_tuples = [(entry[0], unique_map[entry[1]], rc.map_string_to_closest_key(entry[2])) for entry in gold]
            try:
                prediction_tuples = [(entry[0], unique_map[entry[1]], rc.collapse_relationship(entry[2])) for entry in
                                     predict if (entry[0]) in unique_author]
            except:
                prediction_tuples = [(entry[0], unique_map[entry[1]], rc.map_string_to_closest_key(entry[2])) for entry in
                                     predict]
        else:
            gold_tuples = [(entry[0], unique_map[entry[1]], entry[2]) for entry in gold]

            prediction_tuples = [(entry[0], unique_map[entry[1]], entry[2]) for entry in
                                 predict]
        if len(gold_tuples) == 0:
            continue
        if len(prediction_tuples) == 0:
            total_recall.append(0)
            total_precision.append(0)
        else:
            recall, precision = calculate_recall_precision(gold_tuples, prediction_tuples)
            tp, fp, fn = calculate_component(gold_tuples, prediction_tuples)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_recall.append(recall)
            total_precision.append(precision)
    average_macro_recall = sum(total_recall) / len(total_recall)
    average_macro_precision = sum(total_precision) / len(total_precision)
    average_micro_recall = total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0
    average_micro_precision = total_tp / (total_fp + total_tp) if (total_fp + total_tp) != 0 else 0
    if average_micro_precision + average_micro_recall == 0 or average_macro_precision + average_macro_recall == 0:
        return {
            f"MiRec-{mode}": average_micro_recall,
            f"MiPrec-{mode}": average_micro_precision,
            f"MiF1-{mode}": 0,
            f"MaRec-{mode}": average_macro_recall,
            f"MaPrec-{mode}": average_macro_precision,
            f"MaF1-{mode}": 0
        }
    return {
        f"MiRec-{mode}": average_micro_recall,
        f"MiPrec-{mode}": average_micro_precision,
        f"MiF1-{mode}": 2 * average_micro_precision * average_micro_recall / (
                average_micro_precision + average_micro_recall),
        f"MaRec-{mode}": average_macro_recall,
        f"MaPrec-{mode}": average_macro_precision,
        f"MaF1-{mode}": 2 * average_macro_precision * average_macro_recall / (
                average_macro_precision + average_macro_recall)
    }


if __name__ == '__main__':
    gold = [line.strip() for line in open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/author_org/valid_triples.jsonl')]
    # gold = open(
    #     '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/tiny_valid_repaired_resolved_triples.jsonl').readlines()
    gold_triples = [json.loads(line) for line in gold]
    predict = [line.strip() for line in open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/author_org/valid_openai_4omini_legal_ft_triples2.jsonl')]
    predict_triples = [json.loads(line) for line in predict]
    print(evaluate(gold_triples, predict_triples, mode=3))
