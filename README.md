# hadrian_vllm
Take CAD Model of ideal shape, Blueprint with tolerances, return unified standard.

``` # python-occ only on conda
conda activate
```


from OCC.Core.STEPCAFControl import STEPCAFControl_Reader

conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba


To run vertex ai, needed for gemini-2.0
```
gcloud auth application-default login
```


# Order of Files
## Making basic test case from NIST Data
make_basic_test_cases.py and stack_xlsx_files.py, were used to generated csvs which when fixed w/ llm help in https://docs.google.com/spreadsheets/d/1tdtUZaJAdkri25PN4LroGV-EJTjW0Uh98TclfjHdJlk/edit?gid=0#gid=0
created the test data answer keys at `data/fsi_labels/Hadrian Vllm test case - Final Merge.csv`. Gemini 2.0 pro was about the same as O1 in terms of transcribing the special GD&T unicode chars; I had added GD&T chars as part of the prompt. I removed the STR's that were for transcribing notes off the pages from final answer, only want to measure element IDs.

Replaced emoji m with text M.
Removed entries for transcriping full note text on diagram.
deleted rows with `VSI1`, were circles with B,C,D in big letters but not GD&T/PMI data. Not sure what for




