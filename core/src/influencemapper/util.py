# coding: utf-8
# Copyright 2024 Network Dynamics Lab, McGill University
# Distributed under the MIT License

from fuzzywuzzy import process, fuzz


class RelationshipCollapsed:
    """
    A class to collapse the relationship types into a smaller set of categories.
    """

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

    def get_keys(self):
        return list(self.uncollapsed_to_collapsed.keys())

    def collapse_relationship(self, relationship):

        if relationship in self.uncollapsed_to_collapsed:
            return self.uncollapsed_to_collapsed[relationship]
        return relationship

    def map_string_to_closest_key(self, input_string):
        """
        Maps a given string to the closest key in the `uncollapsed_to_collapsed` dictionary using fuzzywuzzy.

        Args:
            input_string (str): The string to map.

        Returns:
            str: The closest key in the `uncollapsed_to_collapsed` dictionary.
            str: The corresponding value from `uncollapsed_to_collapsed` dictionary.
        """
        # Extract all keys from the dictionary
        keys = list(self.get_keys())

        # Find the closest match to the input string
        closest_match, _ = process.extractOne(input_string, keys)

        # Retrieve the corresponding collapsed category
        collapsed_category = self.collapse_relationship(closest_match)

        return collapsed_category

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


def is_similar(str1, str2, threshold=80):
    return fuzz.ratio(str1, str2) > threshold


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



