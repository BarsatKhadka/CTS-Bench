from extract_placement_def_to_dict import process_design , extract_die_area
import random , torch

FILENAME = "picorv32_run_20260107_145745" 

design_data  = process_design(FILENAME, clock_port="clk")
print(len(design_data))


def build_gnn_tensors(design_data):
    # 1. Map gate names to integer IDs (0 to N-1)
    name_to_id = {name: i for i, name in enumerate(design_data.keys())}
    num_nodes = len(design_data)
    
    node_features = []
    edge_index = []
    edge_attr = []
    
    for name, data in design_data.items():
        u = name_to_id[name]
        
        # building Node Features 
        x_coord, y_coord = data['coords']
        is_ff = 1.0 if data['type'] == 'flip_flop' else 0.0
        tc = float(data['toggle_count'])
        node_features.append([x_coord, y_coord, is_ff, tc])
        
        # building Edges & Edge Features ---
        # We look at fan_out to build directed edges: current -> neighbor
        if 'fan_out' in data:
            for neighbor in data['fan_out']:
                if neighbor in name_to_id:
                    v = name_to_id[neighbor]
                    v_data = design_data[neighbor]
                    
                    # Calculate Manhattan Distance for edge attribute
                    v_coords = v_data['coords']
                    dist = abs(x_coord - v_coords[0]) + abs(y_coord - v_coords[1])
                    
                    edge_index.append([u, v])
                    edge_attr.append([dist])
                    
    # 2. Convert to Tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return x, edge_index, edge_attr

x, edge_index, edge_attr = build_gnn_tensors(design_data)

print(f"Node Feature Matrix Shape: {x.shape}")     
print(f"Edge Index Shape: {edge_index.shape}")      
print(f"Edge Attribute Shape: {edge_attr.shape}")   


            

            
# sample_keys = random.sample(list(design_data.keys()), 10)


# for k in sample_keys:
#     print(k, design_data[k])