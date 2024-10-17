import json
import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm
import ast
from author_org.evaluate import get_unique_map, uncollapsed_to_collapsed
from study_org.preprocess.openai_pred_eval import rel_map


def infer_is_company(name):
    name = name.strip()
    keywords = ['university', 'college', 'school', 'program', 'hospital', 'department',
                'agency', 'bureau', 'registry', 'federal', 'government', 'ministry', 'municipal', 'state', 'national']
    keywords += ['universidad', 'colegio', 'escuela', 'programa', 'hospital', 'departamento',
 'agencia', 'oficina', 'registro', 'federal', 'gobierno', 'ministerio', 'municipal', 'estado', 'nacional']
    keywords += ['université', 'collège', 'école', 'programme', 'hôpital', 'département',
    'agence', 'bureau', 'registre', 'fédéral', 'gouvernement', 'ministère', 'municipal', 'état', 'national']
    keywords += ['universität', 'college', 'schule', 'programm', 'krankenhaus', 'abteilung',
    'agentur', 'büro', 'register', 'bundes', 'regierung', 'ministerium', 'kommunal', 'staat', 'national']
    keywords += ['università', 'college', 'scuola', 'programma', 'ospedale', 'dipartimento',
    'agenzia', 'ufficio', 'registro', 'federale', 'governo', 'ministero', 'comunale', 'stato', 'nazionale']
    keywords += ['universiteit', 'college', 'school', 'programma', 'ziekenhuis', 'afdeling',
    'agentschap', 'bureau', 'register', 'federaal', 'overheid', 'ministerie', 'gemeentelijk', 'staat', 'nationaal']
    abbr_name = ['NIH', 'NCI', 'NTP', 'NIEHS', 'NIOSH', 'EPA', 'CDC']
    long_name = ['National Institutes of Health', 'National Cancer Institute', 'National Toxicology Program',
                 'National Institute of Environmental Health Sciences',
                 'National Institute for Occupational Safety and Health', 'Environmental Protection Agency',
                 'Centers for Disease Control and Prevention']
    for keyword in keywords:
        if keyword.upper() in name.upper():
            return 'Unlikely'
        for abbr in abbr_name:
            if abbr.upper() == name.upper():
                return 'Unlikely'
            if '(' + abbr.upper() + ')' in name.upper() or (' ' + abbr.upper() in name.upper() or abbr.upper() + ' ' in name.upper()):
                return 'Unlikely'
        for long in long_name:
            if long.upper() in name.upper():
                return 'Unlikely'
    keywords = ['council', 'academy', 'fund', 'foundation', 'health', 'society', 'union', 'division']
    for keyword in keywords:
        if keyword.upper() in name.upper():
            return 'Possibly'
    if name == 'N/A':
        return 'N/A'
    return 'Likely'


def clean_string(s):
    return s.replace('"', '').encode('latin1', errors='replace').decode('unicode-escape', errors='replace')


if __name__ == '__main__':
    number = '134'
    nlp = English()
    nlp.add_pipe("sentencizer")
    author_org_file = f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/output/author_batch_{number}_output.jsonl'
    study_org_file = f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/output/study_batch_{number}_output.jsonl'

    author_org = [json.loads(line.strip()) for line in open(author_org_file)]
    study_org = [json.loads(line.strip()) for line in open(study_org_file)]

    author_org_datasets = []
    for i, x in enumerate(author_org):
        finish_reason = x['response']['body']['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            author_org_datasets.append({'author_info': []})
            continue
        author_org_datasets.append(json.loads(x['response']['body']['choices'][0]['message']['content']))

    study_org_datasets = []
    for i, x in enumerate(study_org):
        finish_reason = x['response']['body']['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            study_org_datasets.append({'author_info': []})
            continue
        study_org_datasets.append(json.loads(x['response']['body']['choices'][0]['message']['content']))

    data = open(
        f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles{number}.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    datasets = [dataset for dataset in datasets if dataset['coi_statements'] != '']
    columns = ['Pubmed ID', 'Title', 'Author Name', 'Affiliation', 'Affiliation-Industry Support', 'Entity Name', 'Entity-Industry Support',
               'Study Entity Relationship (Yes)', 'Study Entity Relationship (No)', 'Author Entity Relationship (Uncollapsed)', 'Author Entity Relationship (Collapsed)', 'Sentence Entity']
    df = pd.DataFrame(columns=columns)
    df_missing = pd.DataFrame(columns=columns)
    assert len(study_org_datasets) == len(datasets) and len(author_org_datasets) == len(datasets)
    for meta, st, ao in tqdm(zip(datasets, study_org_datasets, author_org_datasets)):
        pubmed_id = meta['pubmed']
        title = meta['title']
        coi_statements = meta['coi_statements'][0]['text'] if type(meta['coi_statements']) is list else meta['coi_statements']
        doc = nlp(coi_statements.replace('\n', ' '))
        orgs_from_st = set([item['org_name'] for item in st['study_info']])
        orgs_from_ao = set(
            [organization['org_name'] for item in ao['author_info'] for organization in item['organization']])
        orgs_from_ao.update(orgs_from_st)
        unique_org_map = get_unique_map(orgs_from_ao)
        unique_org_values = set(unique_org_map.values())

        author_from_meta = set([clean_string(author['name']) for author in meta['authors']])
        author_from_ao = set(
            [clean_string(author['author_name']) for author in ao['author_info']])
        author_from_ao.update(author_from_meta)
        unique_author_map = get_unique_map(author_from_ao,
                                           [clean_string(author['name']) for author in meta['authors']])
        unique_author_values = set(unique_author_map.values())
        author_info_map = {hash(unique_author_map[clean_string(author['name'])]): author['affiliation'] for author in meta['authors']}
        st_map = {unique_org_map[item['org_name']]: [(
            r['relationship_type'], r['relationship_indication']) for r in item['relationships']] for item in
            st['study_info']}
        sents_map = {}
        for org in unique_org_values:
            sents = []
            for sent_org in doc.sents:
                if org in str(sent_org):
                    sents.append(str(sent_org))
            sents_map[org] = ' '.join(sents).replace('\n', ' ')
        for author in ao['author_info']:
            author_name = unique_author_map[
                clean_string(author['author_name'])]
            author_org = {unique_org_map[org['org_name']]: org['relationship_type'] for org in
                          author['organization']}
            if hash(author_name) not in author_info_map:
                continue
            affiliation = author_info_map[hash(author_name)]
            affiliation = 'N/A' if affiliation.strip() == '' else affiliation
            if len(unique_org_values) == 0:
                df_missing = pd.concat([df_missing, pd.DataFrame([[pubmed_id, title, author_name, affiliation, infer_is_company(affiliation), 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']], columns=columns)])
            else:
                for org in unique_org_values:
                    if org in author_org:
                        author_entity_rel = author_org[org]
                    else:
                        author_entity_rel = 'N/A'
                    author_entity_rel = 'N/A' if len(author_entity_rel) == 0 else author_entity_rel
                    study_entity_rel = st_map[org] if org in st_map else ('N/A', 'N/A')
                    pos_rel = [rel[0] for rel in study_entity_rel if rel[1] == 'yes']
                    neg_rel = [rel[0] for rel in study_entity_rel if rel[1] == 'no']
                    pos_rel = ', '.join(pos_rel) if len(pos_rel) > 0 else 'N/A'
                    neg_rel = ', '.join(neg_rel) if len(neg_rel) > 0 else 'N/A'
                    sent_org = sents_map[org]
                    sent_org = meta['coi_statements'] if sent_org.strip() == '' else sent_org
                    try:
                        collapsed_rel = ', '.join(list(set([uncollapsed_to_collapsed[rel] for rel in author_entity_rel]))) if author_entity_rel != 'N/A' else 'N/A'
                    except:
                        collapsed_rel = 'N/A'
                        print(author_entity_rel)
                    df = pd.concat([df, pd.DataFrame([[pubmed_id, title, author_name, affiliation,
                                                       infer_is_company(affiliation), org, infer_is_company(org),
                                                       pos_rel, neg_rel, ', '.join(author_entity_rel) if author_entity_rel != 'N/A' else 'N/A', collapsed_rel,
                                                       sent_org]], columns=columns)])
    # df.to_csv(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/{number}b.csv', header=True, index=False,
    #           sep='\t')
    data = open(
        f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles{number}m.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    for meta in datasets:
        for author in meta['authors']:
            pubmed_id = meta['pubmed']
            title = meta['title']
            author_name = clean_string(author['name'])
            affiliation = author['affiliation']
            affiliation = 'N/A' if affiliation.strip() == '' else affiliation
            df_missing = pd.concat([df_missing, pd.DataFrame([[pubmed_id, title, author_name, affiliation,
                                               infer_is_company(affiliation), 'N/A', infer_is_company('N/A'), 'N/A',
                                               'N/A', 'N/A', 'N/A', 'N/A']], columns=columns)])
    df = pd.concat([df, df_missing])
    # df.to_csv(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/{number}b_missing.csv', header=True, index=False,
    #                   sep='\t')
    df.to_excel(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/{number}c.xls', header=True, index=False,)
