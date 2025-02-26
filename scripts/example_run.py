import os

print(os.getcwd())
import sys

print(sys.path)

import asyncio
from src.main import process_element_id

print(1 / 0)


async def run_example():
    answer, df = await process_element_id(
        text_prompt_path="data/prompts/prompt4.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg2.png",
        element_id="D12",
        model_name="gpt-4o",
        n_shot=2,
        eg_per_img=3,
        examples_as_multiturn=False,
    )

    print(f"Answer for D12: {answer}")
    print(df)

    # Try with multi-turn example format
    answer_mt, df_mt = await process_element_id(
        text_prompt_path="data/prompts/prompt4.txt",
        csv_path="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir="data/eval_on/single_images/",
        question_image="data/eval_on/single_images/nist_ftc_07_asme1_rd_elem_ids_pg2.png",
        element_id=["T11", "D12", "D13", "T12", "D14", "T13", "D15", "T14"],
        model_name="gemini-2.0-flash-001",
        n_shot=5,
        eg_per_img=3,
        examples_as_multiturn=True,
    )

    print(f"Answer for D12 (multi-turn): {answer_mt}")
    print(df_mt)


if __name__ == "__main__":
    asyncio.run(run_example())
