import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np

def clone_graph(G):
    cloned_g = nx.Graph()
    for edge in G.edges():
        cloned_g.add_edge(edge[0], edge[1])
    return cloned_g


def cal_betweeness(G):
    result_dict = {}
    node_list = list(G.nodes())
    
    for edge in G.edges():
        u, v = edge
        result_dict[(u,v)] = 0
    
    for i in range(len(node_list)):
        for j in range(i+1,len(node_list)):
            u = node_list[i]
            v = node_list[j]
            try:
                shortest_path = list(nx.all_shortest_paths(G, u, v))
                for edge in G.edges():
                    tmp = 0
                    for path in shortest_path:
                        if edge[0] in path and edge[1] in path:
                            tmp += 1
                    tmp /= len(shortest_path)
                    result_dict[(edge[0],edge[1])] += tmp
            except:
                pass
    
    return result_dict

def bfs(visited,e_lst,v):
    # bfs
    community = []
    queue = []
    
    queue.append(v)
    visited.append(v)
    community.append(v)
    
    while(len(queue) != 0):
        v = queue.pop(0)
        for edge in e_lst:
            if edge[0] == v and (edge[1] not in visited):
                visited.append(edge[1])
                queue.append(edge[1])
                community.append(edge[1])
                
            if edge[1] == v and (edge[0] not in visited):
                visited.append(edge[0])
                queue.append(edge[0])
                community.append(edge[0])

    return community


# bfs划分社区
def partition(G):
    community_lst = []
    visited = []
    v_lst = G.nodes()
    e_lst = G.edges()
    for v in v_lst:
        if v not in visited:
            community_lst.append(bfs(visited,e_lst,v))
            
    return community_lst


def cal_Q(G,cloned_g):
    edge_num = len(cloned_g.edges())
    Q = 0
    
    # 社区划分
    communities = partition(G)
    
    for community in communities:
        e = 0
        a = 0
        
        # 计算社区i的边占原始网络所有边的比例
        for edge in cloned_g.edges():
            if edge[0] in community and edge[1] in community:
                e += 1
            if edge[0] in community or edge[1] in community:
                a += 1
        e /= edge_num
        a /= edge_num
        
        Q += (e - a * a)
        
    return Q,communities


# 可视化
def showCommunity(G, partition, pos):
    cluster = {}    # 每个节点属于哪个社区
    labels = {}
    for index, item in enumerate(partition):
        for node_id in item:
            labels[node_id] = r'$' + str(node_id) + '$' 
            cluster[node_id] = index 
 
    # 可视化节点
    colors = ['r', 'g', 'b', 'y', 'm']
    for index, item in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=item,
                               node_color=colors[index],
                               node_shape='o',
                               node_size=300,
                               alpha=1)
 
    # 可视化边
    edges = {len(partition): []}
    for link in G.edges():
        # cluster间的边
        if cluster[link[0]] != cluster[link[1]]:
            edges[len(partition)].append(link)
        else:
            # cluster内的边
            if cluster[link[0]] not in edges:
                edges[cluster[link[0]]] = [link]
            else:
                edges[cluster[link[0]]].append(link)
 
    for index, edgelist in edges.items():
        # cluster内
        if index < len(partition):
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color=colors[index])
        # cluster间
        else:
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edgelist,
                                   width=1, alpha=0.8, edge_color='#000000')
 
    # 可视化label
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
 
    plt.axis('off')
    plt.show()


def NMI(partition,G):
    cluster = [0 for _ in range(len(G.nodes()))]
    # 正确结果及其聚类数
    ground_truth = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ground_truth_cluster = 2
    
    for i in range(len(cluster)):
        for j in range(len(partition)):
            if (i + 1) in partition[j]:
                cluster[i] = j + 1

    print(cluster)
    print(ground_truth)
    # 计算互信息
    total = len(cluster)
    cluster = np.array(cluster)
    ground_truth = np.array(ground_truth)
    eps = 1.4e-45
    
    I = 0
    for idx_a in range(len(partition)):
        for idx_b in range(ground_truth_cluster):
            idx_a_occur = np.where(cluster == idx_a + 1)    # 输出满足条件的元素的下标
            idx_b_occur = np.where(ground_truth == idx_b + 1)
            idx_ab_occur = np.intersect1d(idx_a_occur[0],idx_b_occur[0])   # Find the intersection of two arrays.
            px = 1.0 * len(idx_a_occur[0]) / total
            py = 1.0 * len(idx_b_occur[0]) / total
            pxy = 1.0 * len(idx_ab_occur) / total
            I += pxy * math.log(pxy / (px * py) + eps,2)
            
    # 计算交叉熵
    Hx = 0
    for idx_a in range(len(partition)):
        idx_a_occur = len(np.where(cluster == idx_a + 1)[0])
        Hx -= (idx_a_occur / total) * math.log(idx_a_occur / total + eps,2)
        
    Hy = 0
    for idx_b in range(ground_truth_cluster):
        idx_b_occur = len(np.where(ground_truth == idx_b + 1)[0])
        Hy -= (idx_b_occur / total) * math.log(idx_b_occur / total + eps,2)
        
    # 计算标准化互信息
    NMI = 2.0 * I / (Hx + Hy)
    return NMI

def main():
    # 读取gml文件
    G=nx.read_gml("d:/vs/python/community/karate.gml")
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=600, node_color='g', node_shape='.')
    plt.show()
    
    cloned_g = clone_graph(G)
    
    best_partition = [[n for n in G.nodes()]]
    max_Q = 0.0
    
    while(len(G.edges()) != 0):
        # 计算边介数
        betweeness_lst = cal_betweeness(G)

        # 移除最大边介数对应的边
        max_edge = ()
        max_betweeness = -1
        for edge,between_val in betweeness_lst.items():
            if between_val > max_betweeness:
                max_betweeness = between_val
                max_edge = edge
                
        print(max_edge,max_betweeness)
        G.remove_edge(max_edge[0],max_edge[1])
        
        # 计算模块度，记录模块度最大时对应的情况
        cur_Q,cur_partition = cal_Q(G,cloned_g)
        if len(cur_partition) != len(best_partition):
            if cur_Q > max_Q:
                max_Q = cur_Q
                best_partition = cur_partition
    
    print(best_partition)

    # 可视化
    showCommunity(cloned_g, best_partition, pos)
    
    print("NMI is:",NMI(best_partition,cloned_g))
    

if __name__ == '__main__':
    main()
