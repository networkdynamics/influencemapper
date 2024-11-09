import json

from fuzzywuzzy import fuzz
from fuzzywuzzy import process



def map_string_to_closest_key(input_string):
    """
    Maps a given string to the closest key in the `uncollapsed_to_collapsed` dictionary using fuzzywuzzy.

    Args:
        input_string (str): The string to map.

    Returns:
        str: The closest key in the `uncollapsed_to_collapsed` dictionary.
        str: The corresponding value from `uncollapsed_to_collapsed` dictionary.
    """
    # Extract all keys from the dictionary
    keys = list(uncollapsed_to_collapsed.keys())

    # Find the closest match to the input string
    closest_match, _ = process.extractOne(input_string, keys)

    # Retrieve the corresponding collapsed category
    collapsed_category = uncollapsed_to_collapsed[closest_match]

    return collapsed_category


def calculate_recall_precision(gold_tuples, prediction_tuples):
    recall = len(set(gold_tuples) & set(prediction_tuples)) / len(set(gold_tuples))
    precision = len(set(gold_tuples) & set(prediction_tuples)) / len(set(prediction_tuples))
    return recall, precision


def calculate_component(gold_tuples, prediction_tuples):
    tp = len(set(gold_tuples) & set(prediction_tuples))
    fp = len(set(prediction_tuples) - set(gold_tuples))
    fn = len(set(gold_tuples) - set(prediction_tuples))
    return tp, fp, fn


def is_similar(str1, str2, threshold):
    return fuzz.ratio(str1, str2) > threshold


def get_unique_map(names, preset_names='', threshold=70):
    if preset_names == '':
        preset_names = names
    unique_names_map = {name: name for name in names}
    for name in names:
        if unique_names_map[name] == name:
            all_similar_names = []
            for name2 in preset_names:
                if name2 != name and is_similar(name, name2, threshold):
                    all_similar_names.append(name2)
            if len(all_similar_names) > 0:
                name2 = max(all_similar_names)
                unique_names_map[name] = name2
    return unique_names_map


def evaluate(gold_triples, predict_triples, mode=3):
    total_recall = []
    total_precision = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
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
                gold_tuples = [(entry[0], unique_map[entry[1]], uncollapsed_to_collapsed[entry[2]]) for entry in gold]
            except:
                gold_tuples = [(entry[0], unique_map[entry[1]], map_string_to_closest_key(entry[2])) for entry in gold]
            try:
                prediction_tuples = [(entry[0], unique_map[entry[1]], uncollapsed_to_collapsed[entry[2]]) for entry in predict if (entry[0]) in unique_author]
            except:
                prediction_tuples = [(entry[0], unique_map[entry[1]], map_string_to_closest_key(entry[2])) for entry in predict]
        else:
            unique_author = [entry[0] for entry in gold]

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
