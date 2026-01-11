
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


SEMANTIC_TEXTREGION_MAP = {
    "marginalia": "margin",
    "page-number": "pagenr",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
}


COLOR_DICT = {
    "background": (0, 0, 0),    # background
    "image": (45, 255, 0),      # image
    "text": (255, 243, 0),      # text
    "margin": (0, 0, 255),      # margin
    "caption": (255, 100, 243), # caption
    "table": (0, 255, 0),       # table
    "pagenr": (0, 100, 15),     # pagenr
    "header": (255, 0, 0),      # header
    "footer": (255, 255, 100),  # footer
    "line": (0, 100, 255),      # line
}