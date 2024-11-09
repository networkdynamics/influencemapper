def collapse_relationship(relationship):
    uncollapsed_to_collapsed = {
        'None': 'None',
        'Other/Unspecified': 'Other',
        'Speakers\' bureau': 'Received payment/fees/etc. from entity',
        'Speakers bureau': 'Received payment/fees/etc. from entity',
        'Speakers’ bureau': 'Received payment/fees/etc. from entity',
        'Consultant': 'Received payment/fees/etc. from entity',
        'Honorarium': 'Received payment/fees/etc. from entity',
        'Personal fees': 'Received payment/fees/etc. from entity',
        'Former employee of': 'Received payment/fees/etc. from entity',
        'Received travel support': 'Received payment/fees/etc. from entity',
        'Expert testimony': 'Received payment/fees/etc. from entity',
        'Received research grant directly': 'Received research support from entity',
        'Received research grant funds indirectly': 'Received research support from entity',
        'Research Trial committee member': 'Received research support from entity',
        'Received research materials indirectly': 'Received research support from entity',
        'Received research materials directly': 'Received research support from entity',
        'Supported': 'Received research support from entity',
        'Salary support': 'Received research support from entity',
        'Scholarship': 'Received academic support from entity',
        'Fellowship': 'Received academic support from entity',
        'Award': 'Received academic support from entity',
        'Named Professor': 'Received academic support from entity',
        'Holds Chair': 'Received academic support from entity',
        'Equity': 'Direct financial relationship with entity',
        'Employee of': 'Direct financial relationship with entity',
        'Board member': 'Direct financial relationship with entity',
        'Patent license': 'Direct financial relationship with entity',
        'Founder of entity or organization': 'Direct financial relationship with entity',
        'Patent': 'Direct financial relationship with entity',
        'Collaborator': 'Direct financial relationship with entity'
    }
    if relationship in uncollapsed_to_collapsed:
        return uncollapsed_to_collapsed[relationship]
    return relationship


def infer_is_funded(name):
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



