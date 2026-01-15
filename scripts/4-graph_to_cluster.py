from extract_placement_def_to_dict import process_design

FILENAME = "picorv32_run_20260107_145745"

design_data  = process_design(FILENAME, clock_port="clk")


sample = list(design_data.items())[:5]
for key, value in sample:
    print(key, value)




# def graph_to_cluster():
#     node_to_idx = {name: i for i, name in enumerate(design_data.keys())}
#     ff_names = [n for n, d in design_data.items() if d['type'] == 'flip_flop']

#     ff_to_cluster_id = {name: i for i, name in enumerate(ff_names)}

#     #go through each flip-flop and check their jaccard index with other using how similar their fan-out is


#     return ff_names , len(ff_names)


# ff_names, num_ff = graph_to_cluster()
# print(f"Number of Flip-Flops: {num_ff}")
# print("Sample Flip-Flop Names:", ff_names[:10])


