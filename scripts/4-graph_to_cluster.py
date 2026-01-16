from extract_placement_def_to_dict import process_design
from collections import deque
import numpy as np

FILENAME = "picorv32_run_20260107_145745"

design_data  = process_design(FILENAME, clock_port="clk")



#aggregate flops and their one hop neighbors
def form_atomic_clusters(design_data):

    claimed_gates = set()
    atomic_clusters = []

    flops = [k for k, v in design_data.items() if v['type'] == 'flip_flop']
    for i, ff_name in enumerate(flops):
        ff_data = design_data[ff_name]

        #keep track of current cluster members , initialize with the flop itself
        cluster_members = [ff_name]

        queue = deque(ff_data.get('fan_out', []))
        while queue: 
            node_name = queue.popleft()

            if node_name not in design_data: continue
            node_data = design_data[node_name]

            #if the node is a flip-flop , skip it , right now we only want to have logical gates for a independent flop
            if node_data['type'] == 'flip_flop':
                continue

            #If a gate is already claimed by another flop , skip it , we will try to later resolve that
            if node_name in claimed_gates:
                continue

            if node_data['type'] == 'logic':
                claimed_gates.add(node_name)
                cluster_members.append(node_name)

                continue

        #build cluster features from cluster members
        valid_coords = []
        for m in cluster_members:
            m_data = design_data[m]
            if 'coords' in m_data:
                valid_coords.append(m_data['coords'])

        if valid_coords:
            arr = np.array(valid_coords)
            centroid = np.mean(arr, axis=0)
            spread = np.std(arr, axis=0)
        else:
            centroid = np.array([0.0, 0.0])
            spread = np.array([0.0, 0.0])   
        
        gravity_center = design_data[ff_name].get('gravity_center', np.array([0.0, 0.0]))
        gravity_vector = design_data[ff_name].get('gravity_vector', np.array([0.0, 0.0]))
        atomic_clusters.append({
            'id': i,
            'flop_name': ff_name,
            'members': cluster_members,
            'centroid': centroid,
            'spread': spread,
            'gravity_center': gravity_center,
            'gravity_vector': gravity_vector
            
        })



    return atomic_clusters , len(atomic_clusters)


all_clusters , num_clusters = form_atomic_clusters(design_data)
print(f"Number of Atomic Clusters: {num_clusters}")
print("Sample Atomic Clusters:", all_clusters[:5])




# sample = list(design_data.items())[:-10]


# for key, value in sample:
#     print(key, value)




# def graph_to_cluster():
#     node_to_idx = {name: i for i, name in enumerate(design_data.keys())}
#     ff_names = [n for n, d in design_data.items() if d['type'] == 'flip_flop']

#     ff_to_cluster_id = {name: i for i, name in enumerate(ff_names)}

#     #go through each flip-flop and check their jaccard index with other using how similar their fan-out is


#     return ff_names , len(ff_names)


# ff_names, num_ff = graph_to_cluster()
# print(f"Number of Flip-Flops: {num_ff}")
# print("Sample Flip-Flop Names:", ff_names[:10])


