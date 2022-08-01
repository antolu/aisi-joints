CLASS_DEFECT = "DEFECT"
CLASS_OK = "OK"

LABEL_MAP = {CLASS_OK: 1, CLASS_DEFECT: 2}

INV_LABEL_MAP = {v: {"id": v, "name": k} for k, v in LABEL_MAP.items()}
