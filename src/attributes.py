import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://sparql.goldenagents.org/sparql")

def get_birthplace(uri):
    sparql.setQuery(f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>

            SELECT ?birthPlaceName WHERE {{

            GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person ;
                    schema:birthPlace ?birthPlace .
            
                ?birthPlace a schema:Place ;
                    schema:name ?birthPlaceName .
                }} 
            }}
            """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if len(bindings) < 1:
        return None
    else:
        return bindings[0]["birthPlaceName"]["value"]

def get_religion(uri):
    sparql.setQuery(f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>
            PREFIX ecartico: <http://www.vondel.humanities.uva.nl/ecartico/lod/vocab/#>
            SELECT ?religionName WHERE {{
                GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                    <{uri}> a schema:Person ;
                        ecartico:religion ?religionName .
                }}
            }}
            """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if len(bindings) < 1:
        return None
    else:
        return bindings[0]["religionName"]["value"]


def get_occupations(uri):
    sparql.setQuery(f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT   
            (GROUP_CONCAT(DISTINCT ?occupationName; SEPARATOR="; ") AS ?occupations) 

        WHERE {{
            GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person ;
                    schema:hasOccupation [ a schema:Role ;
                                        schema:hasOccupation ?occupation ] .
            
                ?occupation a schema:Occupation ;
                    schema:name ?occupationName .
            
                FILTER(LANG(?occupationName) = 'en')
            }}
            
        }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if len(bindings) < 1:
        return None
    else:
        return bindings[0]["occupations"]["value"]

def get_occupational_address(uri):
    sparql.setQuery(f"""
       PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT
            (GROUP_CONCAT(DISTINCT ?workLocationName; SEPARATOR="; ") AS ?worklocations) 

        WHERE {{
            GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person ;
                    schema:workLocation ?workLocationRole .
                
                ?workLocationRole a schema:Role ;
                    schema:workLocation ?workLocation .
                
                ?workLocation a schema:Place ;
                    schema:name ?workLocationName .
                
                OPTIONAL {{ ?workLocationRole schema:startDate ?startDate . }}
                OPTIONAL {{ ?workLocationRole schema:endDate ?endDate . }}
            }}
        }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if len(bindings) < 1:
        return None
    else:
        return bindings[0]["worklocations"]["value"]

        
def get_birthdate(uri):
    sparql.setQuery(f"""
       PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT ?birthDate WHERE {{
            {{
            GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person ;
                    schema:birthDate ?birthDate .
                    
                FILTER(!(ISURI(?birthDate)))
                
                }}
            }}
            UNION {{
                GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person ;
                    schema:birthDate [ a schema:StructuredValue ;
                                        rdf:value ?birthDate ] .
                }}
            }}
        }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if len(bindings) < 1:
        return None
    else:
        return bindings[0]["birthDate"]["value"]


def get_family_relations(uri):
    sparql.setQuery(f"""
        PREFIX bio: <http://purl.org/vocab/bio/0.1/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>

        SELECT 
            (GROUP_CONCAT(DISTINCT ?parentname; SEPARATOR="; ") AS ?parents)
            (GROUP_CONCAT(DISTINCT ?childname; SEPARATOR="; ") AS ?children) 
            (GROUP_CONCAT(DISTINCT ?spousename; SEPARATOR="; ") AS ?spouses) 

        WHERE {{
            GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
                <{uri}> a schema:Person .
                
                OPTIONAL {{ <{uri}> schema:parent ?parent.
                            ?parent schema:name ?parentname . }}
                OPTIONAL {{ <{uri}> schema:children ?child.
                            ?child schema:name ?childname . }}
                OPTIONAL {{ <{uri}> schema:spouse ?spouse.
                            ?spouse schema:name ?spousename . }} 
                
            }}                  
        }}
        """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    parents = []
    children = []
    spouses = []
    if 'parents' not in bindings[0]:
        parents = None
    else:
        parents = bindings[0]["parents"]["value"]
    if 'children' not in bindings[0]:
        children = None
    else:
        children = bindings[0]["children"]["value"]
    if 'spouses' not in bindings[0]:
        spouses = None
    else:
        spouses = bindings[0]["spouses"]["value"]
    return (parents, children, spouses)


def get_business_relations(uri):
    sparql.setQuery(f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        PREFIX ecartico: <http://www.vondel.humanities.uva.nl/ecartico/lod/vocab/#>

        SELECT  
        (GROUP_CONCAT(DISTINCT ?relatedname; SEPARATOR="; ") AS ?brelations) 
        WHERE {{

        GRAPH <https://data.goldenagents.org/datasets/u692bc364e9d7fa97b3510c6c0c8f2bb9a0e5123b/ecartico_20211014> {{
            <{uri}> a schema:Person ;
                ?relation [ a schema:Role ;
                            ?relation ?relatedPerson ] .
            
            ?relatedPerson a schema:Person;
                schema:name ?relatedname .
                
            ?relation rdfs:subPropertyOf ecartico:hasRelationWith .
            
            }}              
        }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results["results"]["bindings"]
    if "brelations" not in bindings[0]:
        return None
    else:
        return bindings[0]["brelations"]["value"]




def get_attributes(uris):
    attributes = {"uri" : [],
                  "birthplace" : [],
                  "religion" : [],
                  "occupations" : [],
                  "occupational_address": [],
                  "birthdate" : [],
                  "parents" : [],
                  "children" : [], 
                  "spouses" : [],
                  "business_relations": []} 
    n = 0
    total = len(uris)
    for uri in uris:
        n += 1
        print(f"{n} / {total}")
        birthplace = get_birthplace(uri)
        religion = get_religion(uri)
        occupations = get_occupations(uri)
        occupational_address = get_occupational_address(uri)
        birthdate = get_birthdate(uri)
        familyrelations = get_family_relations(uri)
        parents = familyrelations[0]
        children = familyrelations[1]
        spouses = familyrelations[2]
        business_relations = get_business_relations(uri)

        attributes["uri"].append(uri)
        attributes["birthplace"].append(birthplace)
        attributes["religion"].append(religion)
        attributes["occupations"].append(occupations)
        attributes["occupational_address"].append(occupational_address)
        attributes["birthdate"].append(birthdate)
        attributes["parents"].append(parents)
        attributes["children"].append(children)
        attributes["spouses"].append(spouses)
        attributes["business_relations"].append(business_relations)
        
    attribute_df = pd.DataFrame(attributes)
    attribute_df = attribute_df.fillna(value=np.nan)
    attribute_df = attribute_df.replace(r'^\s*$', np.nan, regex=True)
    return attribute_df
