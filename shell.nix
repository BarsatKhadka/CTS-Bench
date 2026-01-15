# ~/CTS-Knob-Aware-placement/shell.nix
{ pkgs ? import <nixpkgs> {} }:

let
  pythonPackages = pkgs.python311Packages;
in
pkgs.mkShell {
  name = "cts-knob-env";
  
  buildInputs = with pkgs; [
    # System libraries
    glibc
    stdenv.cc.cc.lib
    
    # Python and tools
    python3
    pythonPackages.numpy
    pythonPackages.torch
    pythonPackages.scipy
    pythonPackages.pandas
    pythonPackages.matplotlib
    pythonPackages.jupyter
    pythonPackages.pip
    pythonPackages.virtualenv
    
    # Build tools
    gcc
    cmake
    gnumake
  ];

  # Set environment variables
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  
  # Shell hook to setup virtual environment
  shellHook = ''
    echo "=== CTS Knob Environment ==="
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
      echo "Creating virtual environment..."
      python -m venv venv
      source venv/bin/activate
      echo "Installing torch-geometric in virtual environment..."
      pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
    else
      source venv/bin/activate
    fi
    
    echo "Virtual environment activated!"
    echo "Run your script with: python scripts/3-def-dict-to-graph.py"
  '';
}