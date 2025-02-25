import os
from pdf2image import convert_from_path


def split_pdf_into_pages_single_img(pdf_path: str, output_dir: str):
	"""
	Splits the PDF at pdf_path into separate PNGs for each page,
	placing them in output_dir. Each page is saved with a page index in the file name.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir, exist_ok=True)

	# Extract the base filename, e.g. "nist_ftc_06_asme1_rd_elem_ids"
	base_name = os.path.splitext(os.path.basename(pdf_path))[0]

	# Convert PDF to list of PIL images (one per page)
	pages = convert_from_path(pdf_path, dpi=150)  # adjust dpi if needed

	for i, page_img in enumerate(pages):
		# Build a name like nist_ftc_06_asme1_rd_elem_ids_pg1.png
		out_name = f"{base_name}_pg{i+1}.png"
		out_path = os.path.join(output_dir, out_name)
		page_img.save(out_path, "PNG")
		print(f"Saved {out_path}")


if __name__ == ""__main__":
	# single pdf pages
	d = "data/NIST-FTC-CTC-PMI-CAD-models/FTC Definitions/"
	pdf_files = list(sorted([f for f in os.listdir(d) if "elem_ids" in f]))
	for pdf_file in pdf_files:
		split_pdf_into_pages_single_img(d + pdf_file, "data/eval_on/single_images/")
