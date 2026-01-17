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
        'num_of_ff': sum(1 for m in all_members if 'ff' in m.lower()),
        'num_of_logic': sum(1 for m in all_members if 'ff' not in m.lower()),
        'type': 'cluster'
    }

final_clusters = merge_atomic_clusters(all_clusters , raw_edges , dist_limit=0.1 , gravity_alignment_threshold=0.86)
print(f"Total Final Clusters after Merging: {len(final_clusters)}")

# print(merge_candidates)



# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np

# def visualize_cluster_graph(atomic_clusters, edge_list, design_data):
#     fig, ax = plt.subplots(figsize=(12, 12))
#     print(f"--- Visualizing Force Graph ({len(edge_list)} Edges) ---")
    
#     # 1. Plot the Edges (The Forces)
#     # We plot these first so they are in the background
#     for (src_id, dst_id) in edge_list:
#         # Get coordinates of the two cluster centroids
#         c1 = atomic_clusters[src_id]['centroid']
#         c2 = atomic_clusters[dst_id]['centroid']
        
#         # Draw a yellow line between them
#         ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c='#f1c40f', linewidth=0.5, alpha=0.4, zorder=1)

#     # 2. Plot the Nodes (The Clusters)
#     centroids = np.array([c['centroid'] for c in atomic_clusters])
#     ax.scatter(centroids[:, 0], centroids[:, 1], c='#2c3e50', s=10, zorder=2, alpha=0.8)
    
#     # Die Boundary
#     ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
    
#     ax.set_title(f"Cluster Interaction Graph\nNodes: {len(atomic_clusters)} | Edges: {len(edge_list)}")
#     plt.show()

# # --- EXECUTE ---
# visualize_cluster_graph(all_clusters, raw_edges, design_data)


# Note: 'edge_list' is the second return value from your function


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import random

# def visualize_random_high_spread(atomic_clusters, design_data, spread_threshold=0.05, num_to_show=3):
#     """
#     Plots a random selection of 'num_to_show' clusters that violate the spread threshold.
#     Uses dotted lines for clarity.
#     """
#     # 1. Filter to find all 'bad' clusters first
#     bad_clusters = []
#     for cluster in atomic_clusters:
#         max_spread = np.max(cluster['spread'])
#         if max_spread > spread_threshold:
#             bad_clusters.append(cluster)
            
#     total_bad = len(bad_clusters)
#     if total_bad == 0:
#         print(f"No clusters found with spread > {spread_threshold}")
#         return

#     # 2. Randomly select a few to show
#     num_to_pick = min(total_bad, num_to_show)
#     selected_clusters = random.sample(bad_clusters, num_to_pick)
    
#     print(f"--- Visualizing {num_to_pick} Random High-Spread Clusters (out of {total_bad} total) ---")

#     fig, ax = plt.subplots(figsize=(12, 12))
    
#     # Draw Die Boundary
#     ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='#555555', facecolor='#f8f8f8'))
    
#     # Assign unique colors for the selected few for clarity
#     colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e'] # Red, Blue, Green, Purple, Orange

#     for i, cluster in enumerate(selected_clusters):
#         root_name = cluster['flop_name']
#         if design_data[root_name]['coords'] is None: continue
        
#         max_spread = np.max(cluster['spread'])
#         color = colors[i % len(colors)]
        
#         root_x, root_y = design_data[root_name]['coords']
        
#         # --- PLOTTING ---
#         # 1. Draw Dotted Connections
#         gate_xs, gate_ys = [], []
#         for member in cluster['members']:
#             if member == root_name: continue
            
#             if design_data[member].get('coords'):
#                 gx, gy = design_data[member]['coords']
#                 gate_xs.append(gx)
#                 gate_ys.append(gy)
#                 # linestyle=':' makes it dotted
#                 ax.plot([root_x, gx], [root_y, gy], c=color, linewidth=1.5, linestyle=':', alpha=0.6, zorder=10)

#         # 2. Plot Logic Gates
#         if gate_xs:
#             ax.scatter(gate_xs, gate_ys, c=color, s=25, alpha=0.8, zorder=11)
            
#         # 3. Plot Root FF (Larger Star)
#         ax.scatter(root_x, root_y, c='black', s=120, marker='*', edgecolors=color, linewidth=1.5, zorder=12)
        
#         # 4. Label it
#         label_text = f"{root_name}\nSpread: {max_spread:.2f}\nGates: {len(gate_xs)}"
#         ax.text(root_x + 0.01, root_y + 0.01, label_text, fontsize=10, 
#                 bbox=dict(facecolor='white', alpha=0.8, edgecolor=color), zorder=20)

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_title(f"Random High-Spread Inspection ({num_to_pick} selected)")
#     ax.set_xlabel("Die X")
#     ax.set_ylabel("Die Y")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()

# # --- EXECUTE ---
# # Run this multiple times to see different examples
# visualize_random_high_spread(all_clusters, design_data, spread_threshold=0.1, num_to_show=5)
        # 3. Plot Root FF (
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


