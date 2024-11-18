# InfluenceMapper

InfluenceMapper is a python library for extracting disclosure information from scholarly articles. It uses fine-tuned OpenAI's GPT models for the extraction of entities and relationships from the text. The functions included in the library are:
- Extract entities from the text.
- Extract relationships between authors and entities.
- Extract relationships between entities and the study.

## Installation

To install the library, run the following command:

```bash
pip install influencemapper
```

## Training the model

The model is trained on a dataset of scholarly articles. The dataset is available at the `data` folder. To train the model, clone the directory and run the following command:

```bash
python core/src/influencemapper/cli.py fine_tune -train_data data/train.jsonl -valid_data data/valid.jsonl -model_name gpt-4o-mini -threshold 1500 study_org 
python core/src/influencemapper/cli.py fine_tune -train_data data/train.jsonl -valid_data data/valid.jsonl -model_name gpt-4o-mini -threshold 1500 author_org
```

As of the writing of this README, the resulting file has to be uploaded manually to the OpenAI platform to fine-tune the model. The model will be available for use after the fine-tuning process is completed.The `threshold` parameter is used to restrict samples, allowing only those with a maximum token count that meets the training requirements to pass.

## Inferring entities and relationships

To infer entities and relationships from a disclosure text, run the following command:

```bash 
python core/src/influencemapper/cli.py infer -data data/test.jsonl -model_name gpt-4o-mini -API_KEY [API_KEY] study_org
python core/src/influencemapper/cli.py infer -data data/test.jsonl -model_name gpt-4o-mini -API_KEY [API_KEY] author_org
```

To get the results, you have to visit the OpenAI platform and download the results. After dowlnoading the results, you need to combine the results back to the original dataset using the following command:

```bash
python core/src/influencemapper/cli.py combine -data data/test.jsonl -result batch*.jsonl study_org
python core/src/influencemapper/cli.py combine -data data/test.jsonl -result batch*.jsonl author_org
```