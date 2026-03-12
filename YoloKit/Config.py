
LINE_CLASS_MAP = {
    "line" : 0
}

PHOTI_CLASS_MAP = {
    "line": 0,
    "image": 1,
    "caption": 2,
    "margin": 3
}


MODERN_CLASS_MAP = {
    "line" : 0,
    "image" : 1,
    "header": 2,
    "footer": 3,
}

# adapted class numbers to HEADER_FOOTER_MAP_V2 
HEADER_FOOTER_MAP = {
    "header": 0,
    "paragraph": 4,
    "page-number": 1,
}

HEADER_FOOTER_MAP_ULTRALYTICS = {
    "header": 0,
    "footer": 1,
    "footnote": 2,
    "page_number": 3,
    "Text area": 4,
    "image": 5
}

SEMANTIC_TEXTREGION_MAP = {
    "marginalia": "margin",
    "page-number": "pagenr",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
}


COLOR_DICT = {
    "background": (0, 0, 0),
    "image": (45, 255, 0),
    "text": (255, 243, 0),
    "margin": (0, 0, 255),
    "caption": (255, 100, 243),
    "table": (0, 255, 0),
    "pagenr": (0, 100, 15),
    "header": (255, 0, 0),
    "footer": (255, 255, 100),
    "line": (0, 100, 255),
}
