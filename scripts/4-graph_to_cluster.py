import random
from extract_placement_def_to_dict import process_design
from collections import defaultdict, deque
import numpy as np
import networkx as nx

FILENAME = "picorv32_run_20260107_145745"

design_data  = process_design(FILENAME, clock_port="clk")


#aggregate flops and their one hop neighbors
def form_atomic_clusters(design_data):

    claimed_gates = {}
    atomic_clusters = []

    raw_edges = set()

    flops = [k for k, v in design_data.items() if v['type'] == 'flip_flop']

    # Shuffle strictly to remove systematic bias
    random.shuffle(flops)

    flop_name_to_id = {name: idx for idx, name in enumerate(flops)}

    for ff_name , i in flop_name_to_id.items():
        ff_data = design_data[ff_name]

        #keep track of current cluster members , initialize with the flop itself
        cluster_members = [ff_name]

        queue = deque(ff_data.get('fan_out', []))
        while queue: 
            node_name = queue.popleft()

            if node_name not in design_data: continue
            node_data = design_data[node_name]

            #if the node is a flip-flop , then current cluster is going to have an edge to that flop because they are related
            if node_data['type'] == 'flip_flop':
                neighbor_id = flop_name_to_id[node_name]
                if i != neighbor_id:
                    edge = tuple(sorted((i, neighbor_id)))
                    raw_edges.add(edge)
                continue
            

            #If a gate is already claimed by another flop , skip it , we will make a link with that flop through an edge
            if node_name in claimed_gates:
                owner_id = claimed_gates[node_name]

                if owner_id != i:
                    edge = tuple(sorted((i, owner_id)))
                    raw_edges.add(edge)
                continue
                

            if node_data['type'] == 'logic':
                claimed_gates[node_name] = i
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
            RuntimeError(f"No valid coordinates found for cluster rooted at {ff_name}")
        
        gravity_center = design_data[ff_name].get('gravity_center', np.array([0.0, 0.0]))
        gravity_vector = design_data[ff_name].get('gravity_vector', np.array([0.0, 0.0]))
        control_net = ff_data.get('control_net', 'NO_RESET')
        atomic_clusters.append({
            'id': i,
            'flop_name': ff_name,
            'members': cluster_members,
            'centroid': centroid,
            'spread': spread,
            'gravity_center': gravity_center,
            'gravity_vector': gravity_vector, 
            'size': len(cluster_members),
            'control_net': control_net
        })



    return atomic_clusters , len(atomic_clusters) , raw_edges


all_clusters , num_clusters , raw_edges = form_atomic_clusters(design_data)


def merge_atomic_clusters(atomic_clusters , raw_edges , dist_limit=0.1 , gravity_alignment_threshold=0.86):
    final_clusters = []
    merge_candidates = defaultdict(list)
    
    # 1. TRAVERSE & FILTER
    for c in atomic_clusters:
        # Check Spread 
        is_high_spread = np.max(c['spread']) > 0.05
        
        if is_high_spread:
            # High spread? It goes directly to final 
            final_clusters.append(c)
        else:
            # Low spread? Add to the bin for its Reset Net
            # This drastically reduces search space , later we will only merge within same reset net
            net = c['control_net']
            merge_candidates[net].append(c)

    print(f"Filtered {len(final_clusters)} High-Spread Clusters.")
    print(f"Processing {len(merge_candidates)} Reset Groups...")

    merged_count = 0
    for net_name, cluster_list in merge_candidates.items():
        # Skip if only 1 cluster in this bin
        if len(cluster_list) < 2:
            final_clusters.extend(cluster_list)
            continue

        used = [False] * len(cluster_list)
        for i in range(len(cluster_list)):
                if used[i]: continue
                
                # Start a new merged group with 'i'
                current_group = [cluster_list[i]]
                used[i] = True
                
                # Look for partners for 'i'
                base_centroid = cluster_list[i]['centroid']
                base_vector = cluster_list[i]['gravity_vector']
                
                # Normalize base vector for dot product (avoid div by zero)
                norm_base = np.linalg.norm(base_vector)
                if norm_base > 0:
                    #unit vector
                    base_vector_u = base_vector / norm_base
                else:
                    base_vector_u = np.zeros(2)

                for j in range(i + 1, len(cluster_list)):
                    if used[j]: continue

                    target = cluster_list[j]
                    # --- CHECK 1: MANHATTAN DISTANCE ---
                    dist = np.sum(np.abs(base_centroid - target['centroid']))

                    if dist > dist_limit:
                        continue

                    # --- CHECK 2: GRAVITY ALIGNMENT (Cosine Similarity) ---
                    target_vector = target['gravity_vector']
                    norm_target = np.linalg.norm(target_vector)
                
                    # If either has no gravity (0,0), we assume they are compatible (neutral)
                    if norm_base > 0 and norm_target > 0:
                        target_vector_u = target_vector / norm_target
                    # Dot product of unit vectors = Cosine Similarity (-1 to 1)
                        alignment = np.dot(base_vector_u, target_vector_u)
                    
                    # If alignment is low, they are pulling apart.
                        if alignment < gravity_alignment_threshold:
                            continue

                # PASSED ALL CHECKS: MERGE
                    current_group.append(target)
                    used[j] = True
                    merged_count += 1

                # Create the final merged object from 'current_group'
                # (Logic to combine centroids, members, etc.)
                final_clusters.append(create_macro_cluster(current_group))

    print(f"Physics Merge Complete. Total Merges: {merged_count}")
    print(f"Final Macro Clusters: {len(final_clusters)}")
    return final_clusters          



def create_macro_cluster(group):
    """ Helper to combine a list of atomic clusters into one Macro Cluster """
    if len(group) == 1:
        return group[0] # No change
        
    all_members = []
    all_centroids = []
    
    # We pick the ID of the first one as the new ID (or generate new one)
    leader_id = group[0]['id']
    reset_net = group[0]['control_net']
    
    for c in group:
        all_members.extend(c['members'])
        all_centroids.append(c['centroid'])
        
    # Recalculate Centroid
    arr = np.array(all_centroids)
    new_centroid = np.mean(arr, axis=0)
    
    # Recalculate Spread (Radius)
    # Note: simple std dev of centroids is an approximation, but fast
    new_spread = np.std(arr, axis=0) 
    
    return {
        'id': leader_id,
        'members': all_members,
        'centroid': new_centroid,
        'spread': new_spread,
        'size': len(all_members),
        'control_net': reset_net,
        'num_of_ff': sum(1 for m in all_members if design_data[m]['type'] == 'flip_flop'),
        'num_of_logic': sum(1 for m in all_members if design_data[m]['type'] == 'logic'),
        'type': 'cluster'
    }

final_clusters = merge_atomic_clusters(all_clusters , raw_edges , dist_limit=0.05 , gravity_alignment_threshold=0.9)
print(f"Total Final Clusters after Merging: {len(final_clusters)}")
randfinal = random.sample(final_clusters , 1)
print(randfinal)

# print(merge_candidates)



# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import random

# def visualize_atomic_clusters(atomic_clusters, design_data, num_samples=5):
#     """
#     Visualizes a random sample of atomic clusters to verify grouping.
#     - STAR = Flip-Flop (The Cluster Root)
#     - DOTS = Logic Gates (The Cluster Members)
#     - LINE = Connection
#     """
#     # Pick random clusters to visualize
#     if len(atomic_clusters) < num_samples:
#         samples = atomic_clusters
#     else:
#         samples = random.sample(atomic_clusters, num_samples)

#     print(f"--- Visualizing {len(samples)} Sample Atomic Clusters ---")

#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     # Draw Die Boundary (0,0 to 1,1)
#     ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='#f0f0f0'))

#     # distinct colors for different clusters
#     colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#e67e22']

#     for i, cluster in enumerate(samples):
#         color = colors[i % len(colors)]
        
#         # 1. Get Root Flip-Flop Coordinates
#         root_name = cluster['flop_name']
#         if root_name not in design_data or 'coords' not in design_data[root_name]:
#             continue
            
#         root_pos = design_data[root_name]['coords']
        
#         # 2. Get Member Gates Coordinates
#         gate_positions = []
#         for member in cluster['members']:
#             if member == root_name: continue # Skip root
#             if member in design_data and 'coords' in design_data[member]:
#                 gate_positions.append(design_data[member]['coords'])

#         # --- PLOTTING ---
        
#         # A. Draw Lines from Root to Gates (Spider legs)
#         for gx, gy in gate_positions:
#             ax.plot([root_pos[0], gx], [root_pos[1], gy], c=color, alpha=0.5, linewidth=1)

#         # B. Draw Logic Gates (Small Dots)
#         if gate_positions:
#             g_arr = np.array(gate_positions)
#             ax.scatter(g_arr[:, 0], g_arr[:, 1], c=color, s=20, marker='o', label=f"Cluster {cluster['id']} Gates")

#         # C. Draw Root Flip-Flop (Big Star)
#         ax.scatter(root_pos[0], root_pos[1], c=color, s=150, marker='*', edgecolors='black', label=f"FF: {root_name}")
        
#         # D. Draw Centroid (X)
#         cx, cy = cluster['centroid']
#         ax.scatter(cx, cy, c='black', s=50, marker='x', alpha=0.7)

#     ax.set_xlim(-0.05, 1.05)
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_title(f"Atomic Cluster Inspection ({len(samples)} Random Samples)")
#     ax.set_xlabel("Normalized Die X")
#     ax.set_ylabel("Normalized Die Y")
#     ax.legend(loc='upper right', fontsize='small')
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()

# # --- RUN THIS BLOCK ---
# # Pass the 'all_clusters' you generated in Phase 1

# import matplotlib.patches as patches
# import numpy as np
# import random

# def visualize_atomic_clusters(atomic_clusters, design_data, num_samples=5):
#     """
#     Visualizes a random sample of atomic clusters to verify grouping.
#     - STAR = Flip-Flop (The Cluster Root)
#     - DOTS = Logic Gates (The Cluster Members)
#     - LINE = Connection
#     """
#     # Pick random clusters to visualize
#     if len(atomic_clusters) < num_samples:
#         samples = atomic_clusters
#     else:
#         samples = random.sample(atomic_clusters, num_samples)

#     print(f"--- Visualizing {len(samples)} Sample Atomic Clusters ---")

#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     # Draw Die Boundary (0,0 to 1,1)
#     ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='#f0f0f0'))

#     # distinct colors for different clusters
#     colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#e67e22']

#     for i, cluster in enumerate(samples):
#         color = colors[i % len(colors)]
        
#         # 1. Get Root Flip-Flop Coordinates
#         root_name = cluster['flop_name']
#         if root_name not in design_data or 'coords' not in design_data[root_name]:
#             continue
            
#         root_pos = design_data[root_name]['coords']
        
#         # 2. Get Member Gates Coordinates
#         gate_positions = []
#         for member in cluster['members']:
#             if member == root_name: continue # Skip root
#             if member in design_data and 'coords' in design_data[member]:
#                 gate_positions.append(design_data[member]['coords'])

#         # --- PLOTTING ---
        
#         # A. Draw Lines from Root to Gates (Spider legs)
#         for gx, gy in gate_positions:
#             ax.plot([root_pos[0], gx], [root_pos[1], gy], c=color, alpha=0.5, linewidth=1)

#         # B. Draw Logic Gates (Small Dots)
#         if gate_positions:
#             g_arr = np.array(gate_positions)
#             ax.scatter(g_arr[:, 0], g_arr[:, 1], c=color, s=20, marker='o', label=f"Cluster {cluster['id']} Gates")

#         # C. Draw Root Flip-Flop (Big Star)
#         ax.scatter(root_pos[0], root_pos[1], c=color, s=150, marker='*', edgecolors='black', label=f"FF: {root_name}")
        
#         # D. Draw Centroid (X)
#         cx, cy = cluster['centroid']
#         ax.scatter(cx, cy, c='black', s=50, marker='x', alpha=0.7)

#     ax.set_xlim(-0.05, 1.05)
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_title(f"Atomic Cluster Inspection ({len(samples)} Random Samples)")
#     ax.set_xlabel("Normalized Die X")
#     ax.set_ylabel("Normalized Die Y")
#     ax.legend(loc='upper right', fontsize='small')
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()

# # --- RUN THIS BLOCK ---
# # Pass the 'all_clusters' you generated in Phase 1
# visualize_atomic_clusters(all_clusters, design_data, num_samples=480)



# def graph_to_cluster():
#     node_to_idx = {name: i for i, name in enumerate(design_data.keys())}
#     ff_names = [n for n, d in design_data.items() if d['type'] == 'flip_flop']

#     ff_to_cluster_id = {name: i for i, name in enumerate(ff_names)}

#     #go through each flip-flop and check their jaccard index with other using how similar their fan-out is


#     return ff_names , len(ff_names)


# ff_names, num_ff = graph_to_cluster()
# print(f"Number of Flip-Flops: {num_ff}")
# print("Sample Flip-Flop Names:", ff_names[:10])


