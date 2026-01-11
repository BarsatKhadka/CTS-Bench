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