from pathlib import Path
import headroom.headroom as pkg

def test_no_explicit_harm_recipes():
    bad=["step-by-step guide to building a bomb","recipe for ricin"]
    for path in Path(pkg.__file__).parent.glob("*.py"):
        text=path.read_text().lower()
        for b in bad:
            assert b not in text
