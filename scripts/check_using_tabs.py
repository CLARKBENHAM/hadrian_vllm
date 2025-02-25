# %%
import os


def check_indentation(file_path):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("    "):  # Check if line starts with 4 spaces
                return False  # File uses spaces
    return True  # File uses tabs or has no indentation


# List of files to check
files_to_check = [
    "./scripts/pdf_to_screenshots.py",
    "./scripts/elem_ids_per_img_quarters.py",
    "./scripts/elem_ids_per_img.py",
    "./scripts/make_basic_test_cases.py",
    "./scripts/stack_xlsx_files.py",
    "./scripts/blueprint_from_pmi_step_ds.py",
    "./data/gd_t_symbols.py",
    "./src/model_caller.py",
    "./src/image_cost.py",
    "./src/prompt_generator.py",
    "./src/munging.py",
    "./src/evaluation.py",
    "./src/cache.py",
    "./src/result_processor.py",
    "./src/utils.py",
]


def convert_tabs_to_spaces(file_path, spaces=4):
    with open(file_path, "r") as file:
        content = file.read()

    # Replace tabs with spaces
    content = content.replace("\t", " " * spaces)

    with open(file_path, "w") as file:
        file.write(content)


print("Files still using spaces for indentation:")
for file_path in files_to_check:
    if os.path.exists(file_path):
        if not check_indentation(file_path):
            print("Fixing", file_path)
            convert_tabs_to_spaces(file_path)
            print(check_indentation(file_path))
    else:
        print(f"File not found: {file_path}")

print("\nCheck complete.")
