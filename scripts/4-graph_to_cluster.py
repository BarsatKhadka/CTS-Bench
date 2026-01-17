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
            final_clusters.append(create_macro_cluster([c]))
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
            final_clusters.append(create_macro_cluster(cluster_list))
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

    atomic_ids = [c['id'] for c in group]  
    
    return {
        'id': leader_id,
        'atomic_ids': atomic_ids,
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

# def visualize_macro_centroids(macro_clusters, title_suffix=""):
#     """
#     Plots a single dot for the centroid of each macro cluster.
#     - Size determines 'mass' (number of gates).
#     - Color determines 'function' (control/reset net).
#     """
#     num_clusters = len(macro_clusters)
#     print(f"--- Visualizing Centroids for {num_clusters} Macro Clusters ---")

#     fig, ax = plt.subplots(figsize=(10, 10))

#     # 1. Draw Die Boundary background
#     ax.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='#333333', facecolor='#f8f9fa'))

#     # Prepare data for scatter plot
#     xs = []
#     ys = []
#     sizes = []
#     colors = []
    
#     # Helper to generate consistent colors for reset nets
#     net_colors = {}
#     def get_net_color(net_name):
#         if net_name not in net_colors:
#             # Generate random bright color
#             net_colors[net_name] = "#" + ''.join([random.choice('3456789ABCDEF') for j in range(6)])
#         return net_colors[net_name]

#     for cluster in macro_clusters:
#         cx, cy = cluster['centroid']
#         xs.append(cx)
#         ys.append(cy)
        
#         # Calculate dot size based on cluster size (using log to keep scale manageable)
#         # Base size 30 + scaled multiplier
#         s = 30 + (np.log1p(cluster['size']) * 25)
#         sizes.append(s)
        
#         # Get color based on control net
#         colors.append(get_net_color(cluster['control_net']))

#     # 2. Plot the Dots (Centroids)
#     scatter = ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10)

#     # Formatting
#     ax.set_xlim(-0.02, 1.02)
#     ax.set_ylim(-0.02, 1.02)
    
#     # Create a custom legend for the reset nets
#     handles = [patches.Patch(color=color, label=net) for net, color in net_colors.items()]
#     # Only show legend if there aren't too many nets
#     if len(handles) <= 10:
#         ax.legend(handles=handles, title="Control Nets", loc='upper right', fontsize='small')

#     ax.set_title(f"GNN Node Representation ({num_clusters} Nodes)\n{title_suffix}")
#     ax.set_xlabel("Normalized Die X")
#     ax.set_ylabel("Normalized Die Y")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.show()

# # --- EXECUTE ---
# # Run this using the 'final_clusters' list you generated in the previous step.
# visualize_macro_centroids(final_clusters, title_suffix="Size=Mass, Color=ResetNet")