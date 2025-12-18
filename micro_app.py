from __future__ import annotations

from datetime import datetime
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Micro Write Test", layout="centered")
st.title("Micro Write Test")

OUT_FILE = Path("./micro_input.txt")

with st.form("write_form", clear_on_submit=False):
    text_input_val = st.text_input("Text input")
    text_area_val = st.text_input("Text area")

    submitted = st.form_submit_button("Save")

if submitted:
    # Determine which field was used
    if text_input_val and not text_area_val:
        source = "st.text_input"
        value = text_input_val
    elif text_area_val and not text_input_val:
        source = "st.text_area"
        value = text_area_val
    elif text_input_val and text_area_val:
        source = "both"
        value = f"[text_input]\n{text_input_val}\n\n[text_area]\n{text_area_val}"
    else:
        source = "none"
        value = ""

    OUT_FILE.write_text(value + "\n", encoding="utf-8")

    st.success(
        f"Wrote {len(value)} chars from {source} "
        f"at {datetime.now().isoformat(timespec='seconds')}"
    )

    st.write("Detected input source:", source)

# Optional debug output
if OUT_FILE.exists():
    st.caption("Current file contents:")
    st.code(OUT_FILE.read_text(encoding="utf-8"), language="text")
