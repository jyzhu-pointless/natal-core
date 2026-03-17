with open("src/natal/simulation_kernels.py", "r") as f:
    text = f.read()

import re
# We only want to remove cache=True from run_tick, run, run_discrete_tick, run_discrete
for func in ["run_tick", "run", "run_discrete_tick", "run_discrete"]:
    pattern = r"(@njit_switch\(cache=True\))\n(def " + func + r"\()"
    text = re.sub(pattern, r"@njit_switch(cache=False)\n\2", text)

with open("src/natal/simulation_kernels.py", "w") as f:
    f.write(text)
