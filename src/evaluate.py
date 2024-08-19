import json


class Evaluate:

    def __init__(self, dataset_path):
        if not dataset_path:
            raise ValueError("Dataset path is required")
        self.dataset = [json.loads(line) for line in open(dataset_path)]
        self.__process_data()

    def __process_data(self):
        # Using a tab separated convention for the gold and prediction files
        for item in self.dataset:
            gold_author_org_rel, pred_author_org_rel, success = prepare_data(item, True)
            print()

    def __calculate_recall_precision(self, actual, predicted):
        true_positive, false_positive, false_negative = self.__calculate_component(actual, predicted)
        recall = true_positive / (true_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)

        return recall, precision

    @staticmethod
    def __calculate_component(actual, predicted, answer_a_classes):
        # correct = 0
        tp = fn = fp = tn = {c: 0 for c in answer_a_classes + ['all']}
        for t, y in zip(actual, predicted):
            # if t == y:
            #     correct += 1
            for a_c in answer_a_classes:
                if y == a_c and t == a_c:
                    tp[a_c] += 1
                    tp['all'] += 1
                if y != a_c and t == a_c:
                    fn[a_c] += 1
                    fn['all'] += 1
                if y == a_c and t != a_c:
                    fp[a_c] += 1
                    fp['all'] += 1
                if y != a_c and t != a_c:
                    tn[a_c] += 1
                    tn['all'] += 1
        # actual_index = [f'{key}-{i + 1}' for key, value in Counter(actual).items() for i in range(value)]
        # predicted_index = [f'{key}-{i + 1}' for key, value in Counter(predicted).items() for i in range(value)]
        #
        # true_positive = len(set(actual_index) & set(predicted_index))
        # false_positive = len(set(predicted_index) - set(actual_index))
        # false_negative = len(set(actual_index) - set(predicted_index))
        return tp, fn, fp

    @staticmethod
    def expand_entries(data):
        result = []

        authors = data.get("author_field", [])
        companies = data.get("company_field", [])
        relationships = data.get("relationship_type_field", [])

        for author in authors:
            for company in companies:
                for relationship in relationships:
                    result.append({
                        "author_field": author,
                        "company_field": company,
                        "relationship_type_field": relationship
                    })

        return result

    # def correct_entries(self, data):
    #     new_data = []
    #     for item2 in data:
    #         if 'author_field' not in item2:
    #             item2['author_field'] = []
    #         if 'company_field' not in item2:
    #             item2['company_field'] = []
    #         if 'relationship_type_field' not in item2:
    #             item2['relationship_type_field'] = []
    #         if type(item2['author_field']) == str:
    #             item2['author_field'] = [item2['author_field']]
    #         if type(item2['company_field']) == str:
    #             item2['company_field'] = [item2['company_field']]
    #         if type(item2['relationship_type_field']) == str:
    #             item2['relationship_type_field'] = [item2['relationship_type_field']]
    #         new_data.extend(self.expand_entries(item2))
    #     return new_data

    def evaluate(self):
        total_recall = []
        total_precision = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for gold_tuple, prediction_tuple in zip(self.gold_tuples, self.prediction_tuples):
            if len(gold_tuple) == 0:
                continue
            if len(prediction_tuple) == 0:
                total_recall.append(0)
                total_precision.append(0)
            else:
                recall, precision = self.__calculate_recall_precision(gold_tuple, prediction_tuple)
                tp, fp, fn = self.__calculate_component(gold_tuple.split('\t'), prediction_tuple.split('\t'))
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_recall.append(recall)
                total_precision.append(precision)
        return {
            "micro_recall": total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0,
            "micro_precision": total_tp / (total_fp + total_tp) if (total_fp + total_tp) != 0 else 0,
            "macro_recall": sum(total_recall) / len(total_recall),
            "macro_precision": sum(total_precision) / len(total_precision)
        }

    # def evaluate(self, dataset, mode=3, relationship_type=AuthorOrgRelType.UNCOLLAPSED, anonymize=True):
    #     # TODO: the dataset format should be clearer
    #     total_recall = []
    #     total_precision = []
    #     total_tp = 0
    #     total_fp = 0
    #     total_fn = 0
    #     failed_dataset = []
    #     collapse = AuthorOrgRelType.get_collapse_method(relationship_type)
    #     for item in dataset:
    #         gold_author_org_rel, pred_author_org_rel, success = prepare_data(item, anonymize)
    #         if not success:
    #             failed_dataset.append(item)
    #         if mode == 1:
    #             gold_tuples = [(entry["author"]) for entry
    #                            in
    #                            gold_author_org_rel]
    #             prediction_tuples = [(entry["author"]) for
    #                                  entry
    #                                  in pred_author_org_rel]
    #         elif mode == 2:
    #             gold_tuples = [(entry["author"], entry["organization"]) for entry in
    #                            gold_author_org_rel]
    #             prediction_tuples = [(entry["author"], entry["organization"]) for entry
    #                                  in pred_author_org_rel]
    #         else:
    #             gold_tuples = [(entry["author"], entry["organization"], collapse(entry['relationship']))
    #                            for entry in gold_author_org_rel]
    #             prediction_tuples = [(entry["author"], entry["organization"], collapse(entry['relationship']))
    #                                  for entry in pred_author_org_rel]
    #         if len(gold_tuples) == 0:
    #             continue
    #         if len(prediction_tuples) == 0:
    #             total_recall.append(0)
    #             total_precision.append(0)
    #         else:
    #             recall, precision = self.calculate_recall_precision(gold_tuples, prediction_tuples)
    #             tp, fp, fn = self.calculate_component(gold_tuples, prediction_tuples)
    #             total_tp += tp
    #             total_fp += fp
    #             total_fn += fn
    #             total_recall.append(recall)
    #             total_precision.append(precision)
    #    average_macro_recall = sum(total_recall) / len(total_recall)
    #     average_macro_precision = sum(total_precision) / len(total_precision)
    #     average_micro_recall = total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0
    #     average_micro_precision = total_tp / (total_fp + total_tp) if (total_fp + total_tp) != 0 else 0
    #     if average_micro_precision + average_micro_recall == 0 or average_macro_precision + average_macro_recall == 0:
    #         return {
    #             f"MiRec-{mode}": average_micro_recall,
    #             f"MiPrec-{mode}": average_micro_precision,
    #             f"MiF1-{mode}": 0,
    #             f"MaRec-{mode}": average_macro_recall,
    #             f"MaPrec-{mode}": average_macro_precision,
    #             f"MaF1-{mode}": 0
    #         }, failed_dataset
    #     return {
    #         f"MiRec-{mode}": average_micro_recall,
    #         f"MiPrec-{mode}": average_micro_precision,
    #         f"MiF1-{mode}": 2 * average_micro_precision * average_micro_recall / (
    #                     average_micro_precision + average_micro_recall),
    #         f"MaRec-{mode}": average_macro_recall,
    #         f"MaPrec-{mode}": average_macro_precision,
    #         f"MaF1-{mode}": 2 * average_macro_precision * average_macro_recall / (
    #                     average_macro_precision + average_macro_recall)
    #     }, failed_dataset


if __name__ == '__main__':
    gold = open(
        '/Users/blodstone/Research/instruct_rel_extract/output/author_token_mapping/tiny_valid_repaired_resolved.jsonl').readlines()
    gold_dataset = [json.loads(line) for line in gold]
    predict = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/batch_R2VQRvIBHAElUFxcP1bpqpoY_output.jsonl').readlines()
    predict_dataset = [json.loads(line) for line in predict]
    print()
