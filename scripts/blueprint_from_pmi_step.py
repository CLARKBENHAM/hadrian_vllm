#!/usr/bin/env python
# o3
import sys
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDataStd import TDataStd_Name, TDataStd_String


def extract_pmi_from_step(file_path):
    """
    Loads a STEP file using the XDE interface and attempts to extract PMI data.
    The PMI data is stored in the document's label tree.
    Returns a list of (label, attribute) tuples containing PMI-relevant strings.
    """
    # Initialize the document
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document()
    app.NewDocument(TCollection_ExtendedString("MDTV-Standard"), doc)

    # Create a STEP reader that supports annotations (PMI)
    step_reader = STEPCAFControl_Reader(doc.GetHandle())
    status = step_reader.ReadFile(file_path)
    if status != 0:
        print("Error: Cannot read STEP file:", file_path)
        sys.exit(1)

    # Transfer the data into the document structure
    if not step_reader.Transfer(doc.GetHandle()):
        print("Error: Transfer failed")
        sys.exit(1)

    # Traverse the document labels to find PMI information
    # PMI data is often stored as string attributes on labels.
    # This example collects any label that has a TDataStd_Name or TDataStd_String attribute containing keywords.
    pmi_results = []
    root = doc.Main()
    labels = TDF_LabelSequence()
    root.Root().Dump(labels)  # Dump the label tree (if available)
    nb = labels.Length()
    for i in range(1, nb + 1):
        label = labels.Value(i)
        # Check for a name attribute – sometimes used to tag PMI entities.
        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID(), name_attr):
            name_val = name_attr.Get().ToCString()
            if "PMI" in name_val.upper() or "GD&T" in name_val.upper():
                pmi_results.append(("Name", name_val))
        # Check for string attributes which might contain annotation text
        str_attr = TDataStd_String()
        if label.FindAttribute(TDataStd_String.GetID(), str_attr):
            str_val = str_attr.Get().ToCString()
            # Use simple keyword search; adjust as needed for your data.
            if "tolerance" in str_val.lower() or "PMI" in str_val.upper():
                pmi_results.append(("String", str_val))
    return pmi_results


def main2():
    step_file = (
        "/Users/clarkbenham/hadrian_vllm/data/NIST-PMI-STEP-Files/AP203 with"
        " PMI/nist_ctc_01_asme1_ap203.stp"
    )
    pmi_data = extract_pmi_from_step(step_file)
    if pmi_data:
        print("Extracted PMI data:")
        for kind, data in pmi_data:
            print(f"[{kind}] {data}")
    else:
        print("No PMI data found.")


### New response
#!/usr/bin/env python
import sys
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFApp import XCAFApp_Application
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDataStd import TDataStd_Name, TDataStd_String


def extract_pmi_from_step(file_path):
    """
    Loads a STEP file using the XDE interface and attempts to extract PMI data.
    The PMI data is stored in the document's label tree.
    Returns a list of (label, attribute) tuples containing PMI-relevant strings.
    """
    # Initialize the document
    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document()
    app.NewDocument(TCollection_ExtendedString("MDTV-Standard"), doc)

    # Create a STEP reader that supports annotations (PMI)
    step_reader = STEPCAFControl_Reader(doc.GetHandle())
    status = step_reader.ReadFile(file_path)
    if status != 0:
        print("Error: Cannot read STEP file:", file_path)
        sys.exit(1)

    # Transfer the data into the document structure
    if not step_reader.Transfer(doc.GetHandle()):
        print("Error: Transfer failed")
        sys.exit(1)

    # Traverse the document labels to find PMI information
    # PMI data is often stored as string attributes on labels.
    # This example collects any label that has a TDataStd_Name or TDataStd_String attribute containing keywords.
    pmi_results = []
    root = doc.Main()
    labels = TDF_LabelSequence()
    root.Root().Dump(labels)  # Dump the label tree (if available)
    nb = labels.Length()
    for i in range(1, nb + 1):
        label = labels.Value(i)
        # Check for a name attribute – sometimes used to tag PMI entities.
        name_attr = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID(), name_attr):
            name_val = name_attr.Get().ToCString()
            if "PMI" in name_val.upper() or "GD&T" in name_val.upper():
                pmi_results.append(("Name", name_val))
        # Check for string attributes which might contain annotation text
        str_attr = TDataStd_String()
        if label.FindAttribute(TDataStd_String.GetID(), str_attr):
            str_val = str_attr.Get().ToCString()
            # Use simple keyword search; adjust as needed for your data.
            if "tolerance" in str_val.lower() or "PMI" in str_val.upper():
                pmi_results.append(("String", str_val))
    return pmi_results


def main():
    step_file = (
        "/Users/clarkbenham/hadrian_vllm/data/NIST-PMI-STEP-Files/AP203 with"
        " PMI/nist_ctc_01_asme1_ap203.stp"
    )
    pmi_data = extract_pmi_from_step(step_file)
    if pmi_data:
        print("Extracted PMI data:")
        for kind, data in pmi_data:
            print(f"[{kind}] {data}")
    else:
        print("No PMI data found.")


if __name__ == "__main__":
    main()
