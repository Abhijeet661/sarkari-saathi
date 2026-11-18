"""Streamlit UI.

    streamlit run app/streamlit_app.py
"""

import logging
import sys
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from saathi.pipeline import Saathi  # noqa: E402
from saathi.schemas import UserProfile  # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Sarkari Saathi", page_icon="📄", layout="wide")


SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_notification.pdf"


@st.cache_resource(show_spinner="Loading models...")
def load_saathi() -> Saathi:
    saathi = Saathi.from_settings()
    # so a first-time visitor can try it without hunting for a PDF
    if not saathi.documents():
        if not SAMPLE_PDF.exists():
            from scripts.make_sample_pdf import build

            build(SAMPLE_PDF)
        saathi.ingest(SAMPLE_PDF)
    return saathi


def safe(fn, *args):
    """Run a Gemini-backed call and show a clean message instead of a traceback."""
    try:
        return fn(*args)
    except ValueError as err:  # missing GEMINI_API_KEY, or a scanned PDF with no text
        st.error(str(err))
    except Exception as err:
        if getattr(err, "code", None) in (429, 500, 503, 504) or "Timeout" in type(err).__name__:
            st.warning("Gemini is busy right now (too many requests). Please try again in a minute.")
        else:
            st.error(f"Something went wrong ({type(err).__name__}). Please try again.")
    st.stop()


saathi = safe(load_saathi)

st.title("Sarkari Saathi")
st.caption("Ask anything about a government job notification. Answers come with page numbers so you can verify.")

with st.sidebar:
    st.header("Notifications")
    uploaded = st.file_uploader("Upload a notification PDF", type="pdf")
    if uploaded is not None:
        with st.spinner("Reading PDF..."):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / uploaded.name
                path.write_bytes(uploaded.getvalue())
                new_id = safe(saathi.ingest, path)
                st.success(f"Indexed: {new_id}")

    docs = saathi.documents()
    doc_id = st.selectbox("Choose notification", docs) if docs else None

if not doc_id:
    st.info("Upload a notification PDF from the sidebar to start.")
    st.stop()

tab_ask, tab_details, tab_elig = st.tabs(["Ask", "Key details", "Am I eligible?"])

with tab_ask:
    question = st.text_input("Your question (English / हिंदी / मराठी)", placeholder="What is the age limit for Technical Assistant?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Searching the notification..."):
            ans = safe(saathi.ask, question, doc_id)
        st.markdown(ans.text)
        if ans.found and ans.cited_pages:
            st.caption("Pages: " + ", ".join(str(p) for p in ans.cited_pages))
        with st.expander("Show the parts of the PDF used"):
            for s in ans.sources:
                st.markdown(f"**Page {s.chunk.page}**")
                st.text(s.chunk.text)

with tab_details:
    if st.button("Extract key details"):
        with st.spinner("Reading the whole notification..."):
            st.session_state["details"] = safe(saathi.details, doc_id)
    details = st.session_state.get("details")
    if details:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total posts", sum(p.vacancies or 0 for p in details.posts))
        c2.metric("Last date", str(details.application_end or "-"))
        c3.metric("Fee (General)", f"₹{details.fee_general}" if details.fee_general is not None else "-")
        st.dataframe([p.model_dump() for p in details.posts], use_container_width=True)
        st.json(details.model_dump(mode="json"), expanded=False)

with tab_elig:
    with st.form("profile"):
        dob = st.date_input("Date of birth", value=date(2003, 1, 1), min_value=date(1960, 1, 1))
        category = st.selectbox("Category", ["GEN", "EWS", "OBC", "SC", "ST"])
        pwbd = st.checkbox("Person with Benchmark Disability (PwBD)")
        quals = st.text_area("Your qualifications (one per line)", "B.Tech Computer Science\n12th")
        submitted = st.form_submit_button("Check")
    if submitted:
        profile = UserProfile(
            date_of_birth=dob, category=category, pwbd=pwbd,
            qualifications=[q.strip() for q in quals.splitlines() if q.strip()],
        )
        with st.spinner("Checking..."):
            results = safe(saathi.eligibility, doc_id, profile)
        icons = {"eligible": "✅", "not_eligible": "❌", "check_manually": "⚠️"}
        for r in results:
            st.markdown(f"{icons[r.status]} **{r.post}** — {r.status.replace('_', ' ')}")
            for reason in r.reasons:
                st.caption("• " + reason)
        st.caption("Always confirm with the official notification before applying.")
