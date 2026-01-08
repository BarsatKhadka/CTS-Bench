filename = "picorv32_run_20260107_145745"
design_name = filename.split("_")[0]

def load_def():
    path = f"./runs/{filename}/11-openroad-detailedplacement/{design_name}.def"
    with open(path, "r") as f:
        return f.read()


def_text = load_def()

print(def_text)