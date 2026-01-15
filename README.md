Notes for now 

#Steps to Reproduce
# Run once per design for all designs first. 
We run these variables on that file once per design. Those variables are pre placement , we do so , so that we strictly learn the effects of placement and nothing else.



iverilog -o sim_gate.out \
    -DFUNCTIONAL \
    -DUNIT_DELAY=#1 \
    designs/picorv32/tb/testbench.v \
    runs/picorv32_run_20260107_145745/11-openroad-detailedplacement/picorv32.nl.v \
    designs/primitives.v \
    designs/sky130_fd_sc_hd.v

vvp sim_gate.out +vcd


we normaliize coords of each cell by die area. 
min max normalization
x - x(min) / x(max) - x(min)  , similarly for y 

where x(min) , x(max) are extracted from DIE AREA (x, y) (x,y) of the design


We use log scaling to normalize toggle counts.
prevents high-activity nets from dominating the gradient updates

update flake.nix of your openlane , which i will provide.


Nodes (V): Every standard cell (Logic or Flip-Flop) is a node.

Edges (E): Directed connections from fan_in to the current node, and from the current node to fan_out.

Node Features (X): A vector [x,y,is_ff,toggle_count].

Edge Features (Eattr​): The Manhattan Distance (∣x1​−x2​∣+∣y1​−y2​∣) between connected cells.
manhattan distance because two edges are directed to represent the flow and direction. So absolute value of two coordinates would work fine.


Clustering
We use the Jaccard Index, which is the standard way to calculate the overlap between two sets.
Similarity Score=Union (Total Unique Logic)Intersection (Shared Logic)​

    Intersection: The count of gates that appear in both fan_out lists.

    Union: The count of unique gates across both lists combined.

    80% Threshold: If the score is >0.8, it means they share almost all their logic.


The purpose of clustering is not to make a multibank flipflop even though the principal is based on that. it is to show that those two cells are inseperable  , when you move them you move them togethet

Two more things are added to node 
Gravity Center (The "Destination")

    What is it? An absolute coordinate (X,Y).

    Definition: The mathematical average position of everyone pulling on the flip-flop (Input Gates + Output Gates).

    Meaning: "If this flip-flop were free to move, this is exactly where it would land to minimize wire length."

Gravity Vector (The "Force")

    What is it? A relative distance (ΔX,ΔY).

    Definition: Gravity Center−Current Position.

    Meaning: "Which direction is the flip-flop being pulled, and how hard?"

        Vector (+10,0): "I am being pulled hard to the Right."

        Vector (0,+2): "I am being pulled slightly Up."