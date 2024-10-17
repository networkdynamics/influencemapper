import pandas as pd
import tqdm
from nltk.corpus import stopwords
from numpy.ma.extras import unique

from author_org.evaluate import get_unique_map, is_similar
from postprocess_iarc import infer_is_company


def get_unique_map(names, orig_name, window, threshold=70):
    unique_names_map = {}
    for i, name in tqdm.tqdm(enumerate(names)):
        all_similar_names = []
        start = max(i-window//2, 0)
        end = min(i+window//2, len(names))
        for j in range(start, end):
            if len(names[i]) > len(names[j]):
                x = names[i]
                y = names[j]
            else:
                x = names[j]
                y = names[i]
            if is_similar(x, y, threshold):
                all_similar_names.append(orig_name[j])
            elif f"({x})" in y:
                all_similar_names.append(orig_name[j])
        if len(all_similar_names) > 0:
            name2 = max(all_similar_names)
            unique_names_map[orig_name[i]] = name2
    return unique_names_map

def process_name(name):
    keywords = ['university', 'college', 'school', 'program', 'hospital', 'department',
                'agency', 'bureau', 'registry', 'federal', 'government', 'ministry', 'municipal', 'state', 'national']
    keywords += ['universidad', 'colegio', 'escuela', 'programa', 'hospital', 'departamento',
                 'agencia', 'oficina', 'registro', 'federal', 'gobierno', 'ministerio', 'municipal', 'estado',
                 'nacional']
    keywords += ['université', 'collège', 'école', 'programme', 'hôpital', 'département',
                 'agence', 'bureau', 'registre', 'fédéral', 'gouvernement', 'ministère', 'municipal', 'état',
                 'national']
    keywords += ['universität', 'college', 'schule', 'programm', 'krankenhaus', 'abteilung',
                 'agentur', 'büro', 'register', 'bundes', 'regierung', 'ministerium', 'kommunal', 'staat', 'national']
    keywords += ['università', 'college', 'scuola', 'programma', 'ospedale', 'dipartimento',
                 'agenzia', 'ufficio', 'registro', 'federale', 'governo', 'ministero', 'comunale', 'stato', 'nazionale']
    keywords += ['universiteit', 'college', 'school', 'programma', 'ziekenhuis', 'afdeling',
                 'agentschap', 'bureau', 'register', 'federaal', 'overheid', 'ministerie', 'gemeentelijk', 'staat',
                 'nationaal']
    name = name.split(',')[0]
    name = ' '.join([s for s in name.split() if s.lower() not in stopwords.words('english') and s.lower() not in keywords])
    return name

if __name__ == '__main__':
    file_names = ['120b_missing.xlsx', '120b.xlsx', '131b_missing.xlsx', '131b.xlsx', '134b_missing.xlsx', '134b.xlsx']
    path = '/Users/blodstone/Research/influencemapper/InfluenceMapper/data'
    dfs = []
    for file_name in file_names:
        dfs.append(pd.read_excel(f'{path}/{file_name}', engine='openpyxl', header=1))
    df = pd.concat(dfs, axis=0)
    new_df = pd.concat([df['Entity Name'], df['Affiliation']], axis=0)
    new_df = new_df.dropna()
    x = sorted(list(set(new_df.to_list())))
    x2 = [process_name(c) for c in x]
    y = get_unique_map(x2, x, window=10, threshold=70)
    columns = ['Organization', 'Mapped Organization']
    new_df = pd.DataFrame(columns=columns)
    for key, value in y.items():
        new_df = new_df.append({'Organization': key, 'Mapped Organization': value}, ignore_index=True)
    new_df.to_csv(f'{path}/unique_orgs.csv', index=False, sep='\t')
    print()