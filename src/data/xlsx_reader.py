"""Đọc .xlsx trực tiếp từ XML (không cần openpyxl). Trả về {tên sheet: list các dòng}."""
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path, sheets=None) -> dict:
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(_M + "si"):
            shared.append("".join(t.text or "" for t in si.iter(_M + "t")))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out = {}
    for sh in ET.fromstring(z.read("xl/workbook.xml")).iter(_M + "sheet"):
        name = sh.get("name")
        if sheets and name not in sheets:
            continue
        target = rels[sh.get(_R)].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for _, el in ET.iterparse(z.open(target)):
            if el.tag != _M + "row":
                continue
            row = {}
            for c in el.findall(_M + "c"):
                v = c.find(_M + "v")
                t = c.get("t")
                if t == "s":
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(_M + "t"))
                else:
                    val = v.text if v is not None else None
                row[_col_index(c.get("r"))] = val
            rows.append(row)
            el.clear()
        width = max((max(r) for r in rows if r), default=-1) + 1
        out[name] = [[r.get(i) for i in range(width)] for r in rows]
    return out


def sheet_frame(rows, header_row=0) -> pd.DataFrame:
    """Dòng header_row làm tên cột; bỏ cột không tên và dòng rỗng hoàn toàn."""
    header = rows[header_row]
    keep = [i for i, h in enumerate(header) if h not in (None, "")]
    df = pd.DataFrame([[r[i] if i < len(r) else None for i in keep] for r in rows[header_row + 1:]],
                      columns=[header[i] for i in keep])
    return df.dropna(how="all")
