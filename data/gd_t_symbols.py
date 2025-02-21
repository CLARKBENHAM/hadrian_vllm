# %%

gd_symbols = {
    "gd_t_symbols": [
        "⏤",
        "⏥",
        "○",
        "⌭",
        "⌒",
        "⌓",
        "⟂",
        "∠",
        "∥",
        "⌯",
        "⌖",
        "◎",
        "↗",
        "⌰",
        "↔",
        "↗",
        "↧",
        "∠",
        "∥",
        "⌀",
        "⌒",
        "⌓",
        "⌖",
        "⌢",
        "⌭",
        "⌯",
        "⌰",
        "⌲",
        "⌳",
        "⌴",
        "⌵",
        "⏤",
        "⏥",
        "Ⓕ",
        "Ⓛ",
        "Ⓜ",
        "Ⓟ",
        "Ⓢ",
        "Ⓣ",
        "Ⓤ",
        "□",
        "◎",
        "◯",
        "⟂",
        "",
        "",
    ],
    "modifiers_symbols": [
        "→",
        "⊕",
        "⊖",
        "⌂",
        "⌂",
        "Ⓔ",
        "Ⓡ",
        "△",
        "◊",
        "○",
        "",
        "Ⓕ",
        "Ⓛ",
        "Ⓜ",
        "Ⓟ",
        "Ⓢ",
        "Ⓣ",
        "Ⓤ",
    ],
}
if __name__ == "__main__":
    import json
    import csv
    import string

    # Original code points list
    # from "https://en.wikipedia.org/wiki/Geometric_dimensioning_and_tolerancing"
    # "https://docs.techsoft3d.com/hps/2024/prog_guide/appendix_gdt.html"
    gd_t_hex = [
        "U+23E4",
        "U+23E5",
        "U+25CB",
        "U+232D",
        "U+2312",
        "U+2313",
        "U+27C2",
        "U+2220",
        "U+2225",
        "U+232F",
        "U+2316",
        "U+25CE",
        "U+2197",
        "U+2330",
        "0x2194",
        "0x2197",
        "0x21A7",
        "0x2220",
        "0x2225",
        "0x2300",
        "0x2312",
        "0x2313",
        "0x2316",
        "0x2322",
        "0x232D",
        "0x232F",
        "0x2330",
        "0x2332",
        "0x2333",
        "0x2334",
        "0x2335",
        "0x23E4",
        "0x23E5",
        "0x24BB",
        "0x24C1",
        "0x24C2",
        "0x24C5",
        "0x24C8",
        "0x24C9",
        "0x24CA",
        "0x25A1",
        "0x25CE",
        "0x25EF",
        "0x27C2",
        "0xE400",
        "0xE401",
    ]

    modifiers_hex = [
        "0x2192",
        "0x2295",
        "0x2296",
        "0x2302",
        "0x2302",
        "0x24BA",
        "0x24C7",
        "0x25B3",
        "0x25CA",
        "0x25CB",
        "0xE402",
        "U+24BB",
        "U+24C1",
        "U+24C2",
        "U+24C5",
        "U+24C8",
        "U+24C9",
        "U+24CA",
    ]
    # "Final" answers, with my editing
    gd_t_name_mapping = {
        "⏤": ["Straightness"],
        "⏥": ["Flatness"],
        "○": ["Circularity"],
        "⌭": ["Cylindricity"],
        "⌒": ["Profile of a Line", "Profile"],  # "Profile" in answer key
        "⌓": ["Profile of a Surface", "Profile Surface"],
        "⟂": ["Perpendicularity"],
        "∠": ["Angularity"],
        "∥": [
            "Parallelism"
        ],  # correct unicode but not what it looks like on diagram, maybe just a rendering thing?
        # "//": ["Parallelism"], # TODO needed hack? asem8 T14 T21 T5 T16 T17
        "⌯": ["Symmetry"],
        "⌖": ["Position"],
        "◎": ["Concentricity"],
        "↗": ["Circular Runout"],
        "⌰": ["Total Runout"],
        "⌀": ["Diameter"],
        "⌢": ["Arc"],  # for arc length
        "◯": ["Roundness"],  # (guess: alternate circular symbol)
        "": ["Custom GD&T Symbol 1"],  # (nonstandard; proprietary glyph)
        "": ["Custom GD&T Symbol 2"],  # (nonstandard; proprietary glyph)
        # Not a symbol but used in answer key
        "±": ["Plus or Minus"],
        "…": [".*"],  # used for dot dot dot for rest of notes, don't bother testing on this
        "↔": [
            "Connecting To"
        ],  # ["Flatness Extent Indicator", "Flatness Boundary Indicator"] # not a GD&T symbol, but was part of text to indicate "Defines that flatness applies between line elements L1 and L2"
        "↧": ["Depth"],  # ["Depth of Feature", "Surface Profile"] # not GD&T either
        # what does NIST data call these things though? Doesn't matter since only symbol used
        "⌲": ["Conical Taper"],  # (U+2332)
        "⌳": ["Slope"],  # FFlat Taper  # (U+2333)
        "⌴": ["Counterbore"],  # (U+2334)
        "⌵": ["Countersink"],
        "□": ["Square"],  # (U+25A1)
    }
    modifiers_name_mapping = {
        "→": ["Directional Indicator"],  # (guess: indicates a direction)
        "⊕": ["Additive Tolerance"],  # (guess: plus–type modifier)
        "⊖": ["Subtractive Tolerance"],  # (guess: minus–type modifier)
        "⌂": ["Composite Tolerance"],  # (guess: used on open or composite features)
        # Second occurrence of "⌂" is identical.
        "Ⓔ": ["Envelope Requirement"],
        "Ⓡ": ["Reference Feature"],  # (guess: indicates a referenced feature)
        "△": [
            "Datum Target",
            "Datum Feature",
            "Datum Feature Symbol",
        ],  # (triangle often marks datum targets)
        "◊": ["Basic Dimension"],  # (diamond is used for basic or theoretically exact dimensions)
        "○": [
            "All Around"
        ],  # Circulatiry as symbol, All Around if modifies a bar out. No dedicated all around unicode char? Depends on context? # (in this context, an alternate circular modifier),
        "": ["Custom Modifier 1"],  # (nonstandard custom modifier)
        "Ⓕ": ["Free State"],  # standard modifier
        "Ⓛ": ["Least Material Condition"],
        "Ⓜ": ["Maximum Material Condition"],
        "Ⓟ": ["Projected Tolerance Zone"],
        "Ⓢ": ["Regardless of Feature Size"],
        "Ⓣ": ["Tangent Plane"],
        "Ⓤ": ["Unequal Bilateral"],
    }

    def check_chars_exhaustive(known_symbols):
        # Create a set of all known symbols from our dictionaries

        # Define a set for storing found unique special Unicode characters
        found_special_chars = set()
        chars_used = set()
        # Characters considered common (ASCII letters, digits, whitespace, and basic punctuation)
        common_chars = set(
            string.ascii_letters + string.digits + string.whitespace + string.punctuation
        )

        csv_file = "data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"

        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spec = row.get("Specification", "")
                for char in spec:
                    chars_used.add(char)
                    # If the character is not in our known_symbols
                    if char not in known_symbols:
                        # And if it's not a common ASCII character
                        if char not in common_chars:
                            found_special_chars.add(char)

        print("Unique special Unicode characters in 'Specification' not in our mapping:")
        for char in sorted(found_special_chars):
            # Print the character and its Unicode code point for clarity
            print(f"{char} (U+{ord(char):04X})")
        print("Unused Chars")
        for char in sorted(known_symbols - chars_used):
            # Print the character and its Unicode code point for clarity
            print(f"{char} (U+{ord(char):04X})")

        print("Used Chars")
        print(known_symbols & chars_used)

    # first try
    # check_chars_exhaustive(set(gd_t_symbols.keys()) | set(modifiers_symbols.keys()))
    # Response we go with:
    check_chars_exhaustive(set(modifiers_name_mapping.keys()) | set(gd_t_name_mapping.keys()))

    def code_point_to_char(code_point):
        # Remove 'U+' or '0x' prefix and convert to integer
        if code_point.startswith(("U+", "0x")):
            return chr(int(code_point[2:], 16))
        else:
            return chr(int(code_point))

    # Convert code points to characters
    gd_t_hex_chars = [code_point_to_char(cp) for cp in gd_t_hex]
    modifiers_hex_chars = [code_point_to_char(cp) for cp in modifiers_hex]
    _hex = gd_t_hex + modifiers_hex

    # Create the JSON object
    json_obj = {
        "gd_t_hex": gd_t_hex,
        "modifiers_hex": modifiers_hex,
        **gd_symbols,
        "hex_to_symbol": {h: code_point_to_char(h) for h in _hex},
        "symbol_to_hex": {code_point_to_char(h): h for h in _hex},
        "symbol_to_name": {**gd_t_name_mapping, **modifiers_name_mapping},
    }

    # Serialize to JSON-formatted string
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=4)
    with open("data/gd_t_symbols.json", "w", encoding="utf-8'") as f:
        f.write(json_str)
    print(json_str)
