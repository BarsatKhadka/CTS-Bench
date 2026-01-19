import os
import sys
import json
import glob

# --- 1. Get the Run Tag ---
if len(sys.argv) > 1:
    RUN_TAG = sys.argv[1]
elif os.path.exists("latest_run.txt"):
    RUN_TAG = open("latest_run.txt").read().strip()
else:
    sys.exit("❌ Error: No Run Tag provided and latest_run.txt not found.")

# Base path for this run
RUN_DIR = os.path.join("runs", RUN_TAG)
CTS_EXP_DIR = os.path.join(RUN_DIR, "CTS-experiments")

if not os.path.exists(CTS_EXP_DIR):
    sys.exit(f"❌ Error: CTS experiments not found in {RUN_DIR}")

results = []

print(f"📊 Parsing metrics for run: {RUN_TAG}")

# --- 2. Loop through CTS-1 to CTS-10 ---
# We use glob to find all CTS-* folders
cts_folders = sorted(glob.glob(os.path.join(CTS_EXP_DIR, "CTS-*")))

for folder in cts_folders:
    exp_id = os.path.basename(folder) # e.g., "CTS-1"
    
    # Path to the critical JSON file
    json_path = os.path.join(folder, "2-openroad-stamidpnr", "state_out.json")
    knobs_path = os.path.join(folder, "knobs.json")

    # Skip if files don't exist (maybe run failed)
    if not os.path.exists(json_path):
        print(f"   ⚠️  Skipping {exp_id}: state_out.json not found.")
        continue

    try:
        # Load Knobs (Inputs)
        knobs = {}
        if os.path.exists(knobs_path):
            with open(knobs_path, 'r') as f:
                knobs = json.load(f)

        # Load Metrics (Outputs)
        with open(json_path, 'r') as f:
            data = json.load(f)
            m = data.get("metrics", {})

        # Extract ONLY what we care about
        # Use .get() to avoid crashing if a key is missing
        parsed_metrics = {
            "skew_setup": m.get("clock__skew__worst_setup"),
            "skew_hold":  m.get("clock__skew__worst_hold"),
            "power_total": m.get("power__total"),
            "wirelength": m.get("route__wirelength__estimated"),
            "setup_slack": m.get("timing__setup__ws"),
            "hold_slack": m.get("timing__hold__ws"),
            "clock_buffers": m.get("design__instance__count__class:clock_buffer")
        }

        # Add to list
        results.append({
            "id": exp_id,
            "knobs": knobs,
            "metrics": parsed_metrics
        })

        print(f"   ✅ {exp_id}: Skew={parsed_metrics['skew_setup']}ns | Power={parsed_metrics['power_total']}W")

    except Exception as e:
        print(f"   ❌ {exp_id}: Error parsing JSON - {e}")

# --- 3. Save to File ---
output_file = os.path.join(RUN_DIR, "dataset.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n💾 Saved {len(results)} data points to: {output_file}")