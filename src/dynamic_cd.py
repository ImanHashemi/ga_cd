from operator import truediv
from socket import getnameinfo
import networkx as nx
from cdlib import algorithms
import matplotlib.pyplot as plt
from cdlib import evaluation
from SPARQLWrapper import SPARQLWrapper, JSON
import os
from cdlib.algorithms import louvain
from cdlib import algorithms, viz
from pathlib import Path
import csv
import numpy as np
from cdlib import TemporalClustering




sparql = SPARQLWrapper("https://sparql.goldenagents.org/sparql")

def getName(uri):
    sparql.setQuery (f"""
        SELECT ?s ?t
        WHERE {{ OPTIONAL {{ <{uri}> <http://schema.org/name> ?t}} }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    return list(results["results"]["bindings"][0].values())[0]["value"]

def getBookSet(g, cluster_edges):
    bookset = set([])
    # count = 0

    for edge in cluster_edges:
        # count += len(g[edge[0]][edge[1]]['books'])
        bookset |= g[edge[0]][edge[1]]['books']
      
        
    return (len(bookset), bookset)

# get the earliest timestamp and the latest timestamp of a set of edges of a community
def getMinMaxDates(g, cluster_edges):
    earliestList = []
    latestList = []

    for edge in cluster_edges:
        for year in g[edge[0]][edge[1]]['bt']:
            earliestList.append(year)    
        for year in g[edge[0]][edge[1]]['et']:
            latestList.append(year)

  
    bins = 10
    if len(earliestList) > 11:
        mean_earliest = np.mean(earliestList)
        std_earliest = np.std(earliestList)
        earliest_time = mean_earliest - (2 * std_earliest)
    else:
        earliest_time = min(earliestList)

    if len(latestList) > 11:
        mean_latest = np.mean(latestList)
        std_latest = np.std(latestList)
        latest_time = mean_latest + (2 * std_latest)
    else:
        latest_time = max(latestList)


        
    # plt.hist(earliestList, bins=bins, color='darkblue', edgecolor='white');
    # plt.hist(latestList, bins=bins, color='darkblue', edgecolor='white');
    # plt.show()
    return (int(earliest_time), int(latest_time))
    # return (mean_earliest)


def edgeInCluster(edge, cluster):
    if edge[1] in list(cluster):
        return True
    return False

def getLCC(hg):
    components = nx.connected_components(hg)
    largest_subgraph_size = max(components, key=len)
    lcc = hg.subgraph(largest_subgraph_size)
    return lcc


def create_graph_snapshot(G, start, end):
    G_snapshot = nx.Graph(earliest_time=start, latest_time=end)

    for edge in G.edges():
        # G_snapshot.add_nodes_from([edge[0],edge[1]])
        avg_time = G[edge[0]][edge[1]]['at']
        for time in avg_time:
            if time >= start and time <= end:
                if G_snapshot.has_edge(edge[0], edge[1]):
                    G_snapshot[edge[0]][edge[1]]['weight'] += 1
                    # G_snapshot[edge[0]][edge[1]].add(G[edge[0])][edge[1])]['books'])
                    # G_snapshot[edge[0]][edge[1]].append(G[edge[0])][edge[1])]['bt'])
                    # G_snapshot[edge[0]][edge[1]].append(G[edge[0])][edge[1])]['et'])
                else:
                    G_snapshot.add_edge(edge[0], edge[1], weight=1, books=G[edge[0]][edge[1]]['books'], bt=G[edge[0]][edge[1]]['bt'], 
                    et= G[edge[0]][edge[1]]['et'])
                    

    return G_snapshot




def create_snapshot_graph(G, interval, stepsize):
    earliest = G.graph['earliest_time']
    latest = G.graph['latest_time']
    curr_timestamp = earliest   
    snapshots = []
    snapshot_times = []

    while curr_timestamp < latest:
        start = curr_timestamp
        end = curr_timestamp + interval
        G_snapshot = create_graph_snapshot(G, start, end)
        if (len(G_snapshot.edges()) > 0):            
            snapshots.append(G_snapshot)
            snapshot_times.append((start, end))
        curr_timestamp = start + stepsize


    print(f"Interval: {interval}, Stepsize: {stepsize}")    
    print(f"Num of snapshots: {len(snapshots)}")   
    # for snapshot in snapshots:
        # print(f"Num of edges {len(snapshot.edges)}")
        # print(f"Num of LCC nodes {len(getLCC(snapshot).nodes)}")
    
    # nx.draw_networkx(snapshots
    # [0], with_labels=False, node_size=40)
    return (snapshots, snapshot_times)



def list_to_dict(partition):
    partition_dict = {}
    community_id = 0
    for community in partition:
        for node in community:
            partition_dict[node] =  community_id
        community_id += 1

def dynamic_community_detection(snapshots, snapshot_times, parameters, randomize, use_init_partition, use_init_partition_randomize):
    tc = TemporalClustering()
    last_partition = None
    for index, snapshot in enumerate(snapshots):
        # nx.draw_networkx(snapshots[index], with_labels=False, node_size=40)
        plt.show()
        if randomize:
            best_coms = algorithms.louvain(snapshot, randomize=None)
            best_mod = evaluation.newman_girvan_modularity(snapshot, best_coms).score
            for x in range(10):
                coms = algorithms.louvain(snapshot, randomize=None)  # here any CDlib algorithm can be applied
                mod = evaluation.newman_girvan_modularity(snapshot, coms).score
                print(f"Mod: {mod}")
                if mod > best_mod:
                    best_coms = coms
                    best_mod = mod
            coms = best_coms
        elif use_init_partition:
            if last_partition is not None:
                coms = algorithms.louvain(snapshot, partition=list_to_dict(last_partition.communities))  # here any CDlib algorithm can be applied
                last_partition = coms  
            else:
                coms = algorithms.louvain(snapshot)
                last_partition = coms  # here any CDlib algorithm can be applied
        elif use_init_partition_randomize:
            if last_partition is not None:
                coms = algorithms.louvain(snapshot, partition=list_to_dict(last_partition.communities))  # here any CDlib algorithm can be applied
                last_partition = coms  
            else:
                best_coms = algorithms.louvain(snapshot, randomize=None)
                best_mod = evaluation.newman_girvan_modularity(snapshot, best_coms).score
                for x in range(10):
                    coms = algorithms.louvain(snapshot, randomize=None)  # here any CDlib algorithm can be applied
                    mod = evaluation.newman_girvan_modularity(snapshot, coms).score
                    print(f"Mod: {mod}")
                    if mod > best_mod:
                        best_coms = coms
                        best_mod = mod
                last_partition = best_coms                
        else:
            coms = algorithms.louvain(snapshot)
            
        mod = evaluation.newman_girvan_modularity(snapshot, coms).score
        # print(mod)
        if(mod > 0):
            tc.add_clustering(coms, index)
        # print(f"Done with CD in snapshot {index}")
        
    jaccard = lambda x, y:  len(set(x) & set(y)) / len(set(x) | set(y))
    digraph = tc.lifecycle_polytree(jaccard)
    stability_trend = None
    return (digraph, stability_trend, tc, snapshots, snapshot_times)



def execute_dcd(sgs):
    dcd_results_init_partition = {}

    for key, item in sgs.items():
            print(f"Handling sg with parameters: {key}" )
            dcd_init_partition = dynamic_community_detection(item[0], item[1], key, randomize=False, use_init_partition=True, use_init_partition_randomize=False)
            dcd_results_init_partition[key] = dcd_init_partition
    

    tc = dcd_results_init_partition['20_10'][2]
    snapshots = dcd_results_init_partition['20_10'][3]
    snapshot_times = dcd_results_init_partition['20_10'][4]


    return (dcd_results_init_partition, tc, snapshots, snapshot_times)
        

def matching(tc, theta):
    jaccard = lambda x, y:  len(set(x) & set(y)) / len(set(x) | set(y))
    dynamic_coms = {}
    fronts = {}
    timesteps = tc.get_observation_ids()
    init_clustering = tc.get_clustering_at(timesteps[0]).communities
    uid = 0

    # Create a dynamic community for every cluster at the first timestep and add them to the fronts of the dynamic communities
    for index, comm in enumerate(init_clustering):
        dynamic_coms[uid] = [str(timesteps[0]) + "_" + str(index)]
        fronts[uid] = str(timesteps[0]) + "_" + str(index)
        uid += 1
    
    # For each subsequent timestep, get all communities and match them with all fronts,
    # adding them if their simmilarity is higher than theta
    for t in timesteps[1:]:
        t_clustering = tc.get_clustering_at(t).communities
        for index, comm in enumerate(t_clustering):
            matches = []
            for front_id, front_comm in fronts.items():
                # print(front_comm)
                sim = jaccard(comm, tc.get_community(front_comm))
                # print(sim)
                if sim >= theta:
                    matches.append(front_id)
            if len(matches) == 0:
                dynamic_coms[uid] = [str(t) + "_" + str(index)]
                fronts[uid] = str(t) + "_" + str(index)
                uid += 1
            else:
                if len(matches) > 1:
                    print(f"MERGING {matches}")
                for match in matches:
                    if fronts[match].split()[0] == str(t):
                        print("SPLIT")
                        dynamic_coms[uid] = dynamic_coms[match][:-1].append(str(t) + "_" + str(index))
                        fronts[uid] = str(t) + "_" + str(index)
                        uid += 1
                    else:
                        dynamic_coms[match].append(str(t) + "_" + str(index))
                        fronts[match] = str(t) + "_" + str(index)
    return dynamic_coms

def print_dyn_com(dynamic_coms, id, tc, snapshots, snapshot_times):
    dynamic_com = dynamic_coms[id]
    # print(f"TEST: {dynamic_com}")
    cache_folder = Path("./cache/")    
    file_to_check = cache_folder / "attribute_cache.gpickle"
    is_cached = None
    if not file_to_check.exists():
        is_cached = False
        attribute_cache = {}
    else:
        is_cached = True
        attribute_cache = nx.read_gpickle(cache_folder / "attribute_cache.gpickle")

    for part in dynamic_com:
        part_timestep = part.split('_')[0]
        com_id = part.split('_')[1]
        partition = tc.get_clustering_at(int(part_timestep)).communities
        snapshot = snapshots[int(part_timestep)]
        com = partition[int(com_id)]
        timestep = snapshot_times[int(part_timestep)]
        print(timestep)
        for node in com:
            if node in attribute_cache:
                com_name = attribute_cache[node]
            else:
                com_name = getName(node)
                attribute_cache[node] = com_name
            print(com_name)

        print("-------------")

        # print("COMS: " + str(coms))
        # print("COM_IDS: " + str(filtered_com_ids))

        # snapshot = snapshots[int(part_timestep)]
        # sorted_coms = sort_coms(partition, snapshot)

def match_dyn_com(dynamic_coms, id):
    for dyn_id, coms in dynamic_coms.items():
        if id in coms:
            return dyn_id



def find_dyn_com(dynamic_coms, name, tc, snapshots, snapshot_times):
    timesteps = tc.get_observation_ids()
    t0 = timesteps[0]
    print(f"NAME TO FIND: {name}")
    cache_folder = Path("./cache/")    
    file_to_check = cache_folder / "attribute_cache.gpickle"
    is_cached = None
    if not file_to_check.exists():
        is_cached = False
        attribute_cache = {}
            # print("HG saved in folder \"homogeneous_graphs\"")
    else:
        is_cached = True
        attribute_cache = nx.read_gpickle(cache_folder / "attribute_cache.gpickle")
            # print("HG imported")

    full_com_list = []
    for timestep in timesteps:
        partition = tc.get_clustering_at(timestep).communities
        com_id = None
        # for all communities in timestep
        for index, com in enumerate(partition):
            # print("-----")
            com_index = None
            # for all nodes in community
            for node in com:
                if node in attribute_cache:
                    com_name = attribute_cache[node]
                else:
                    com_name = getName(node)
                    attribute_cache[node] = com_name
                # print(com_name)
                if com_name == name:
                    com_index = index
                    break
            if com_index is not None:
                com_id = com_index
                break
        if com_id is not None:
            full_com_id = str(timestep) + '_' + str(com_id)
            full_com_list.append(full_com_id)
            print(full_com_id)

    for full_com_id in full_com_list:
        print(f"TIMESTEP: {full_com_id.split('_')[0]}")

        dyn_com_id = match_dyn_com(dynamic_coms, full_com_id)
        print(f"DYNAMIC COMMUNITY: {dyn_com_id}")
        partition = tc.get_clustering_at(int(full_com_id.split('_')[0])).communities

        com = partition[int(full_com_id.split('_')[1])]
        timestep = snapshot_times[int(full_com_id.split('_')[0])]
        print(f"INTERVAL {timestep}")
        for node in com:
            if node in attribute_cache:
                com_name = attribute_cache[node]
            else:
                com_name = getName(node)
                attribute_cache[node] = com_name
            print(com_name)
        print("---------")
        # print_dyn_com(dy
        # 
        # namic_coms, dyn_com_id, tc, snapshots, snapshot_times)
        # print(match_dyn_com(dynamic_coms, dyn_com_id))


        # print("COMS: " + str(coms))
        # print("COM_IDS: " + str(filtered_com_ids))

        # snapshot = snapshots[int(part_timestep)]
        # sorted_coms = sort_coms(partition, snapshot)

# find_dyn_com(dynamic_coms, "Jan Jansz. Veenhuysen", tc, snapshots, snapshot_times)

























def community_detection_louvain(g):
    g_lcc = getLCC(g)

    # Perform the louvain commiunity detecion algorithm on the homogeneous graph
    louvain_communities = louvain(g)
    mod_louvain = louvain_communities.newman_girvan_modularity()
    avg_embeddedness_louvain = evaluation.avg_embeddedness(g,louvain_communities)
    
    # Perform the louvain commiunity detecion algorithm on the LCC of the homogeneous graph
    louvain_communities_lcc = louvain(g_lcc)
    mod_louvain_lcc = louvain_communities_lcc.newman_girvan_modularity()
    avg_embeddedness_louvain_lcc = evaluation.avg_embeddedness(g_lcc,louvain_communities_lcc)

    # Print some analysis on the resulting clustering
    print(f"Number of communities (louvain): {len(louvain_communities.communities)}")
    print(f"Number of communities (louvain_lcc): {len(louvain_communities_lcc.communities)}")
    print()
    print(f"Modularity (louvain): {mod_louvain}")
    print(f"Modularity (louvain lcc): {mod_louvain_lcc}")
    print()
    print(f"Average Embeddedness (louvain): {avg_embeddedness_louvain}")
    print(f"Average Embeddedness (louvain lcc): {avg_embeddedness_louvain_lcc}")

    # Draw the communities
    fig = plt.figure(1)
    viz.plot_network_clusters(g, node_size=150, partition=louvain_communities)
    viz.plot_network_clusters(g_lcc, node_size=150, partition=louvain_communities_lcc)
    plt.show()

    return (louvain_communities, louvain_communities_lcc)

def format_partition(g, coms):
    # For each community, add all nodes to a dict along with their centrality measure,
    # sort this dict based on centrality, and add the dict to a list of communities
    dc_sorted_communities_list = []
    degree_centrality = nx.degree_centrality(g)
    cache_folder = Path("./cache/")    
    file_to_check = cache_folder / "attribute_cache.gpickle"
    is_cached = None
    if not file_to_check.exists():
        is_cached = False
        attribute_cache = {}
            # print("HG saved in folder \"homogeneous_graphs\"")
    else:
        is_cached = True
        attribute_cache = nx.read_gpickle(cache_folder / "attribute_cache.gpickle")
            # print("HG imported")

    for cluster in coms.communities:
        dc_centrality_dict = {}
        cluster_edges = []
        for node in cluster:
            edges = g.edges(node)
            for edge in edges:
                if edgeInCluster(edge, cluster):
                    cluster_edges.append(edge)
            if node in attribute_cache:
                name = attribute_cache[node]
            else:
                name = getName(node)
                attribute_cache[node] = name
            # print(name)
            # g[edge[0]][edge[1]]['books']
            dc_centrality_dict[node] = (degree_centrality[node], name)

        dc_centrality_dict_sorted = sorted(dc_centrality_dict.items(), key=lambda item: item[1], reverse=True)
        min_max_dates = getMinMaxDates(g, 
        cluster_edges)
        (bookcount, bookset) =  getBookSet(g, cluster_edges)
        # print(f"Number of books:  {bookcount}")
        print(f"Number of books (length):  {len(bookset)}")

        for book in bookset:
            if book.uri in attribute_cache:
                book_name = attribute_cache[book.uri]
            else:
                book_name = getName(book.uri)
                attribute_cache[book.uri] = book_name
            print(book_name)

        dc_sorted_communities_list.append((dc_centrality_dict_sorted, min_max_dates, bookset, bookcount))
    if not is_cached:
        nx.write_gpickle(attribute_cache, cache_folder / "attribute_cache.gpickle")
    return (dc_sorted_communities_list, attribute_cache)


def export(sorted_communities, metapath_name, attribute_cache):
    data_folder = Path("results/")

    # Export topx community list
    filename =  metapath_name + "_expo.txt"
    file_to_write = data_folder / filename
    os.makedirs(os.path.dirname(file_to_write), exist_ok=True)
    with open(file_to_write, mode='w') as communities:
        for index1, community in enumerate(sorted_communities):
            communities.write(f"Community {index1+1}, total members of the community: {len(community[0])}, Period: {community[1]}, Number of books: {community[3]}")
            communities.write('\n')
            communities.write("Top 5 nodes according to degree centrality:")
            communities.write('\n')
            top5 = list(community[0])[:5]
            for index2, person in enumerate(top5):
                uri = person[0]
                cm = person[1][0]
                name = person[1][1]
                communities.write(f"{index2} : {name}, CM: {round(cm, 3)} (URI: {uri})")
                communities.write('\n')

            communities.write('\n')
            communities.write('\n')
        print(f"File {metapath_name}_expo.txt' created.")

    # Export full community list
    filename =  metapath_name + ".csv"
    file_to_write = data_folder / filename
    with open(file_to_write, mode='w') as communities:
        fieldnames = ["Community", "Name", "Centrality Measure", "URI", "Community Period"]
        coms_writer = csv.writer(communities, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        coms_writer.writerow(fieldnames)

        for index1, community in enumerate(sorted_communities):
            # print(community)
            for index2, person in enumerate(community[0]):
                # print(person)
                uri = person[0]
                cm = person[1][0]
                name = person[1][1]
                period = community[1]
                # print(f"{index2} : {name}, DC: {cm} (URI: {uri})")
                # print(f"{index2} : {name}, DC: {n[1]} (URI: {n[0]})")
                coms_writer.writerow([index1 + 1, name, round(cm, 3), uri, period])
        print(f"File {metapath_name}.csv' created.")

    
    # Export community attributes list
    filename =  metapath_name + "_attributes.csv"
    file_to_write = data_folder / filename
    os.makedirs(os.path.dirname(file_to_write), exist_ok=True)
    with open(file_to_write, mode='w') as communities:
        coms_writer = csv.writer(communities, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index1, community in enumerate(sorted_communities):
            coms_writer.writerow([f"Community {index1+1} - Period: ({community[1][0]} - {community[1][1]} - Number of books: {community[3]}"])
            # coms_writer.writerow()
            # communities.writerow("Books:")
            # communities.writerow('\n')
            
            coms_writer.writerow(["Name", "URI", "Begin", "End"])
            for book in community[2]:
                coms_writer.writerow([attribute_cache[book.uri], book.uri, book.bt, book.et])

        print(f"File {metapath_name}_attributes.csv' created.")