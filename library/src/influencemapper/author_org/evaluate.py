# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License

import json

from influencemapper.util import RelationshipCollapsed, get_unique_map

def format_to_tuples(line):
    data = json.loads(line)
    triples = [(author_data['__name'], org[0][0] if type(org[0]) is list else org[0], org[1]) for _, author_data in
                data['author_info'].items() for org in author_data['__relationships']]
    return triples

def get_entities(data):
    pass

def get_entities_authors_pairs(data):
    pass

def get_relationships_tuple(data):
    pass

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
