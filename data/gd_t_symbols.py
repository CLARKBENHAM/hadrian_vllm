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
        **{h: code_point_to_char(h) for h in _hex},
        **{code_point_to_char(h): h for h in _hex},
    }

    # Serialize to JSON-formatted string
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=4)
    with open("data/gd_t_symbols.json", "w", encoding="utf-8'") as f:
        f.write(json_str)
    print(json_str)
