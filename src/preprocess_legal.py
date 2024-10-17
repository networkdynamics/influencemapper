import json
import ast
import pandas as pd
if __name__ == '__main__':
    dataset = pd.read_csv('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/legal.tsv', sep='\t')
    data = open(
        '/Users/blodstone/Research/influencemapper/InfluenceMapper/data/valid.jsonl').readlines()
    datasets = [json.loads(line) for line in data]
    header = ['annotator_id', 'year_x', 'session_id', 'article_id', 'article_title', 'author_info', 'confusing_issue',
              'entities', 'study_info', 'tech_issue', 'author_count', 'class', '_id', 'scopus_link', 'abstract',
              'journal_document_type', 'title', 'citations', 'scraper_data', 'research_article', 'coden', 'journal',
              'original_language', 'source_type', 'authors', 'document_type', 'coi_statements', 'journal_standardized',
              'doi', 'full_text_html_url', 'full_html', 'issn', 'publisher', 'coi_scraper_error', 'mdata_scraper_error',
              'journal_link', 'alt_doi', 'num_pages']
    new_dataset = []
    for index, row in dataset.iterrows():
        authors = [author for author in ast.literal_eval(row['Authors'])]
        coi_statements = row['COI']
        title = row['Title']
        pubmed = row['PubMed']
        author_org = json.loads(row['fmt_author-org'])
        author_info_line = {}
        for a in author_org:
            author = a['Author']
            entity = a['Entity']
            relationships = [key for key, item in a['Conflicts'].items() if item]
            author_info_line[hash(author)] = {
                '__declared_coi': False,
                '__name': author,
                '__relationships': [[entity, relationship, True] for relationship in relationships]
            }
        data = {
            'authors': [{'name': author, 'affiliation': ''} for author in authors],
            'pubmed': pubmed,
            'title': title,
            'coi_statements': coi_statements,
            'author_info': author_info_line
        }
        new_dataset.append(json.dumps(data))
    open('/Users/blodstone/Research/influencemapper/InfluenceMapper/data/legal.jsonl', 'w').writelines('\n'.join(new_dataset))