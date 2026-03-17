with open("src/natal/hook_dsl.py") as f:
    text = f.read()
idx = text.find("        # Sort by priority and compile combined functions")
print("Found at:", idx)
