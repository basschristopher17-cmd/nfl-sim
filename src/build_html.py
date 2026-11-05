"""Inject output/slate.json into template.html -> docs/index.html (one self-contained file)."""
import json
from config import ROOT, OUT, DOCS

def main():
    tpl = (ROOT / "template.html").read_text()
    data = (OUT / "slate.json").read_text()
    assert "/*__DATA__*/null" in tpl, "placeholder missing in template.html"
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(tpl.replace("/*__DATA__*/null", data))
    print("wrote", DOCS / "index.html")

if __name__ == "__main__":
    main()
