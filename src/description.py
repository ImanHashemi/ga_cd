from itertools import count
import pandas as pd
import numpy as np


# Get the union of all nodes for all communities within a dynamic community
def get_community_union(tc, dynamic_coms, dynamic_com_id):
    dynamic_com = dynamic_coms[dynamic_com_id]
    union_community = [] 

    for part in dynamic_com:
        part_timestep = part.split('_')[0]
        com_id = part.split('_')[1]
        partition = tc.get_clustering_at(int(part_timestep)).communities
        com = partition[int(com_id)]
        union_community += com
        
    return union_community

# Given a community (list of nodes), get a simple description for it using ndoe attributes
def get_community_description(community, attributes_df):
    # Get attributes for community member
    filtered_attribute_df = attributes_df.loc[attributes_df['uri'].isin(community)]
    
    count_birthplace = filtered_attribute_df['birthplace'].value_counts(normalize=True)
  
    count_religion = attributes_df['religion'].value_counts(normalize=True)
    count_occupations = unpack_attribute(filtered_attribute_df['occupations']).value_counts(normalize=True)
    count_occaddresses = unpack_attribute(filtered_attribute_df['occupational_address']).value_counts(normalize=True)
    
    fam = list(filtered_attribute_df['parents']) + list(filtered_attribute_df['children']) + list(filtered_attribute_df['spouses'])
    family_df = pd.DataFrame(fam, columns=['family']).replace('nan', np.nan).dropna()
    count_family = unpack_attribute(family_df['family']).value_counts(normalize=True)
    
    birthdate_df = pd.DataFrame(filtered_attribute_df["birthdate"].map(lambda x: int(x.split('-')[0]) if type(x) == str else x), columns=['birthdate'])

    if len(count_birthplace) > 0:
        top_birthplace = (count_birthplace.index[0], round(count_birthplace[0] * 100, 2))
    else:
        top_birthplace = "No Data"
    if len(count_religion) > 0:
        top_religion = (count_religion.index[0], round(count_religion[0] * 100, 2))
    else:
        top_religion = "No Data"
    if len(count_occupations) > 0:
        top_occupations = (count_occupations.index[0], round(count_occupations[0] * 100, 2))
    else:
        top_occupations = "No Data"
    if len(count_occaddresses) > 0:
        top_occaddresses = (count_occaddresses.index[0], round(count_occaddresses[0] * 100, 2))
    else:
        top_occaddresses = "No Data"
    if len(count_family) > 0:
        top_family = (count_family.index[0], round(count_family[0] * 100, 2))
    else:
        top_family = "No Data"
    if len(birthdate_df.dropna()['birthdate']) > 0:
        median_birthdate = round(birthdate_df.dropna()['birthdate'].median())    
    else:
        median_birthdate = "No Data"

    description = {
        "Birthplace": top_birthplace,
        "Religion" : top_religion,
        "Occupation" : top_occupations,
        "Occupational Address" : top_occaddresses,
        "Family Relation" : top_family,
        "Birthday" : median_birthdate
    }

    return description

def unpack_attribute(att_df):
    att_list = []
    for row in att_df:
        atts = str(row).split(';')
        att_list.extend(atts)
    return pd.Series(att_list)