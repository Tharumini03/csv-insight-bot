import os

def build_knowledge_file(
    file_id,
    schema,
    cleaning_report,
    model_info,
    feature_importance,
    insight_text
):

    output_dir = f"storage/outputs/{file_id}"
    os.makedirs(output_dir, exist_ok=True)

    knowledge_path = f"{output_dir}/knowledge.txt"

    with open(knowledge_path, "w") as f:

        f.write("DATASET SUMMARY\n")
        f.write(str(schema))
        f.write("\n\n")

        f.write("CLEANING REPORT\n")
        f.write(str(cleaning_report))
        f.write("\n\n")

        f.write("MODEL INFORMATION\n")
        f.write(str(model_info))
        f.write("\n\n")

        f.write("FEATURE IMPORTANCE\n")
        f.write(str(feature_importance))
        f.write("\n\n")

        f.write("INSIGHTS\n")
        f.write(str(insight_text))
        f.write("\n")

    return knowledge_path