import ast
import pickle
import sys
from typing import Dict, List

import pandas as pd
import json

from jinja2.utils import missing

if __name__ == '__main__':
    # df = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/new_inclusionArticles120.tsv', sep='\t')
    number = '134'
    df = pd.read_excel(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles{number}.xlsx', engine='openpyxl')
    new_dataset = []
    missing_dataset = []

    def clean(s):
        return s.replace("\'text\'", '\"text\"').replace("\'cite_spans\'", '\"cite_spans\"').replace("\'ref_spans\'", '\"ref_spans\"').replace("\'eq_spans\'", '\"eq_spans\"').replace("\'section\'", '\"section\"').replace('\\\'', '\'').replace('None', "null")
    cx = 0
    count_not_empty = 0
    cq = 0
    title_set = set()
    for index, row in df.iterrows():
        if not pd.isna(row['COI']) and row['COI'].startswith('EXISTS:'):
            coi_statement = row['Manual']
            cx += 1
        else:
            coi_statement = ''
        assert pd.isna(coi_statement) == False
        cleaned_authors = []
        try:
            json_lines = [json.loads(line) for line in open(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/Vol{number}_json/' + row['PDF'].replace('.pdf', '.jsonl2').replace('\'', '_')).readlines()]
            for author in json_lines[0]['authors']:
                first_name = author['first']
                middle_names = ' '.join(author['middle'])
                last_name = author['last']
                full_name = f"{first_name} {middle_names} {last_name}".strip()
                affiliation = author.get('affiliation', {})
                affiliation_name = ''
                if len(affiliation) > 0:
                    institution = affiliation.get('institution', '')
                    country = affiliation.get('location', {}).get('country', '')
                    affiliation_name = f"{institution}, {country}".strip(', ')
                cleaned_authors.append({
                    'name': full_name.strip(),
                    'affiliation': affiliation_name,
                    'email': author.get('email', '')
                })
            if coi_statement == '':
                coi_statement = json_lines[0]['disclosure']
        except:
            if row['Authors'].startswith('['):
                for x in ast.literal_eval(row['Authors']):
                    if x.strip() != 'et al':
                        cleaned_authors.append({'name': x.strip(), 'affiliation': '', 'email': ''  })
            elif row['Authors'].startswith('\''):
                for x in row['Authors'].replace('\'', '').split(','):
                    if x.strip() != 'et al':
                        cleaned_authors.append({'name': x.strip(), 'affiliation': '', 'email': ''  })
            else:
                for x in row['Authors'].split(','):
                    if x.strip() != 'et al':
                        cleaned_authors.append({'name': x.strip(), 'affiliation': '', 'email': ''  })
        data = {
            'title': row['Title'],
            'authors': cleaned_authors,
            'coi_statements': coi_statement,
            'pdf': row['PDF'],
            'pubmed': row['PubMed']
        }
        if coi_statement == '':
            cq += 1
            missing_dataset.append(json.dumps(data))
        if coi_statement != '' and hash(row['Title']) not in title_set:
            count_not_empty += 1
            title_set.add(hash(row['Title']))
            new_dataset.append(json.dumps(data))
    print(count_not_empty)
    print(cx)
    print(cq)
    open(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles{number}.jsonl', 'w').writelines('\n'.join(new_dataset))
    open(f'/Users/blodstone/Research/influencemapper/InfluenceMapper/data/authors_InclusionArticles{number}m.jsonl',
         'w').writelines('\n'.join(missing_dataset))