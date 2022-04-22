
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from src import attributes

sparql = SPARQLWrapper("https://sparql.goldenagents.org/sparql")

# Create a homogeneous graph based on a given metapath, using the goldenagents endpoint
def rdf_to_homogeneous_endpoint(df, names_df):
    print("Constructing homogeneous graph...")
    first = df.iloc[0]
    avg_time = int((int(first['et'].split('-')[0]) + int(first['bt'].split('-')[0])) / 2)
    earliest_time = avg_time
    latest_time = avg_time
    G = nx.Graph(earliest_time=avg_time, latest_time=avg_time)
    print(f"Number of meta-path instances: {len(df)}")

    # Create a homogeneous graph given a dataframe of meta-path instances
    for index, row in df.iterrows():
        w1 = row["w1"]
        w2 = row['w2']     
        avg_time = int((int(row['et'].split('-')[0]) + int(row['bt'].split('-')[0])) / 2)

        if avg_time < earliest_time:
            earliest_time = avg_time
        elif avg_time > latest_time:
            latest_time = avg_time

        if (G.has_edge(w1, w2)):
            G[w1][w2]['weight'] += 1
            G[w1][w2]['books'].append(row['b'])
            G[w1][w2]['et'].append(row['et'].split('-')[0])
            G[w1][w2]['bt'].append(row['bt'].split('-')[0])
            G[w1][w2]['at'].append(avg_time)
        else:
            G.add_edge(w1, w2, weight=1, books= [row['b']], bt=[row['bt'].split('-')[0]], et=[row['et'].split('-')[0]], at= [avg_time])

    print(f"Num of edges {len(G.edges)}")
    print(f"Num of nodes {len(G.nodes)}")
    G.graph['earliest_time'] = earliest_time
    G.graph['latest_time'] = latest_time
    names_dict = dict(zip(names_df.uri, names_df.name))
    nx.set_node_attributes(G, names_dict, name="name")
    return G    
    

# Pefrom meta-path query and construct a dataframe with
# all data needed to create the homogenous graph
def construct_dataframe(query):
    # Perform query
    print("Performing query...")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    query_results = sparql.queryAndConvert()
    structured_results = query_results["results"]["bindings"]

    print("Constructing dataframe...")
    df = pd.DataFrame(structured_results)
    df = df.applymap(lambda x: x['value'])
    
    # Get the booknames for all 
    booknames = get_names(df['b'].drop_duplicates())
    booknames_df = pd.DataFrame(booknames, columns=["b", "bookname"])

    # Add booknames to the graph dataframe by merging
    df = pd.merge(df, booknames_df, how="inner")

    # Get person names and add them to the graph dataframe by merging
    names = list(df['w1']) + list(df['w2'])
    names_df = pd.DataFrame(names, columns=['person']).drop_duplicates()
    person_names = get_names(names_df['person'])
    w1_df = pd.DataFrame(person_names, columns=["w1", "name1"])
    w2_df = pd.DataFrame(person_names, columns=["w2", "name2"])
    df = pd.merge(df, w1_df, how="inner")
    df = pd.merge(df, w2_df, how="inner")

    # Create a seperate dataframe solely for names
    names_df = pd.DataFrame(person_names, columns=["uri", "name"])

    return (df, names_df)


def init_data(metapath, metapath_queries):
    # Check if the homogeneous graph is present in cache
    # If not, create it
    df_folder = Path("./data/dataframes")
    hg_folder = Path("./data/homogeneous_graphs")

    df_to_check = df_folder / (metapath + ".pkl")
    hg_to_check = hg_folder / (metapath + ".gpickle")

    if not df_to_check.exists():
        query = metapath_queries[metapath]
        (df, names_df) = construct_dataframe(query)
        attributes_df = attributes.get_attributes(names_df['uri'])
        df.to_pickle(df_folder / (metapath + ".pkl"))  
        names_df.to_pickle(df_folder / (metapath + "_names.pkl"))
        attributes_df.to_pickle(df_folder / (metapath + "_attributes.pkl"))  
        print("Dataframes saved in folder \"dataframes\"")
    else:
        df = pd.read_pickle(df_folder / (metapath + ".pkl"))
        names_df = pd.read_pickle(df_folder / (metapath + "_names.pkl"))
        attributes_df = pd.read_pickle(df_folder / (metapath + "_attributes.pkl"))
        print("Dataframes imported")

    if not hg_to_check.exists():
        hg = rdf_to_homogeneous_endpoint(df, names_df)
        nx.write_gpickle(hg, hg_folder / (metapath + ".gpickle"))
        print("HG saved in folder \"homogeneous_graphs\"")
    else:
        hg = nx.read_gpickle(hg_folder / (metapath + ".gpickle")) 
        print("HG imported")

    names_dict = dict(zip(names_df.uri, names_df.name))
    nx.set_node_attributes(hg, names_dict, name="name")

    return (df, names_df, attributes_df, hg)


def get_names(df):
    names = []
    n = 0
    for uri in df.drop_duplicates():
        n += 1
        print(n)
        sparql.setQuery (f"""
                SELECT DISTINCT ?t
                WHERE {{ OPTIONAL {{ <{uri}> <http://schema.org/name> ?t}} }}
            """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        names.append((uri, list(results["results"]["bindings"][0].values())[0]["value"]))
    return names



    