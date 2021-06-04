import dgl
from dgl import DGLGraph
import torch as th
import numpy as np
import tqdm
from graph_info import hetero_graph, drug_index_id_map, microbe_index_id_map


def parse_trace(trace, drug_index_id_map, microbe_index_id_map):
    s = []
    for index in range(trace.size):
        if index % 2 == 0:
            s.append(drug_index_id_map[trace[index]])
        else:
            s.append(microbe_index_id_map[trace[index]])
    return ','.join(s)


def main():
    meta_path = ['dm', 'md', 'dm', 'md', 'dm', 'md', 'dm', 'md']
    num_walks_per_node = 1
    f = open("../output/aBiofilm_output_path.txt", "w")
    for drug_idx in tqdm.trange(hetero_graph.number_of_nodes('drug')):  # 以drug开头的meta_path
        traces = dgl.contrib.sampling.metapath_random_walk(
            hg=hetero_graph, etypes=meta_path, seeds=[drug_idx, ], num_traces=num_walks_per_node)
        tr = traces[0][0].numpy()
        tr = np.insert(tr, 0, drug_idx)
        res = parse_trace(tr, drug_index_id_map, microbe_index_id_map)
        f.write(res + '\n')
    f.close()


if __name__ == '__main__':
    main()
