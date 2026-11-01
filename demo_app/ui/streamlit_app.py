"""
Streamlit front end for the API (api/main.py).

    streamlit run ui/streamlit_app.py

Tabs
    Try it          one recording: zero-shot vs fine-tuned vs pause-guided LLM, word timeline, regions
    Evaluate a set  several recordings with references: corpus WER/CER/MER per system, LLM sign test
    Paper results   the paper's evaluation (Tables 1–3, Fig. 2) and per-speaker breakdowns
"""
import html
import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
SYSTEMS = ["Zero-shot Whisper-tiny", "Fine-tuned Whisper-tiny", "+ pause-guided LLM"]
SYSTEM_KEYS = {"zero_shot": SYSTEMS[0], "finetuned": SYSTEMS[1], "pause_llm": SYSTEMS[2]}

st.set_page_config(page_title="Pauses as Error Cues in Dysarthric ASR", layout="wide")


# ── chart styling (reference palette; validated for both themes) ────────────────
def _theme():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


THEME = _theme()
P = {
    "light": dict(series=["#2a78d6", "#eb6834", "#1baf7a"], good="#2a78d6", bad="#e34948", neutral="#898781",
                  text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#ffffff"),
    "dark": dict(series=["#3987e5", "#d95926", "#199e70"], good="#3987e5", bad="#e66767", neutral="#898781",
                 text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", surface="#0e1117"),
}[THEME]
SYSTEM_SCALE = alt.Scale(domain=SYSTEMS, range=P["series"])
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def show(chart, height=300):
    chart = (chart.properties(height=height)
             .configure(background="transparent", font=FONT)
             .configure_view(strokeWidth=0)
             .configure_axis(gridColor=P["grid"], domainColor=P["axis"], tickColor=P["axis"],
                             labelColor=P["muted"], titleColor=P["text2"], labelFontSize=11, titleFontSize=12,
                             titleFontWeight="normal")
             .configure_axisX(grid=False)
             .configure_legend(labelColor=P["text2"], titleColor=P["text2"], orient="top", title=None,
                               labelFontSize=12, symbolType="circle"))
    st.altair_chart(chart, width="stretch", theme=None)


def table_view(df, label="Table view"):
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


def bar_kwargs():
    # rounded data end and a 2px surface gap between adjacent bars
    return dict(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, stroke=P["surface"], strokeWidth=2)


# thin grouped bars: wide gaps between categories, bars of one category side by side
GROUP_X = alt.Scale(paddingInner=0.55, paddingOuter=0.3)


def pct_axis(title):
    return alt.Axis(format="%", title=title, tickCount=5)


def metric_bars(df, x, metric, x_title, x_sort=None):
    """Grouped bars: one bar per system within each x category."""
    return alt.Chart(df).mark_bar(**bar_kwargs()).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=x_sort, scale=GROUP_X, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("system:N", sort=SYSTEMS),
        y=alt.Y(f"{metric}:Q", axis=pct_axis(metric)),
        color=alt.Color("system:N", scale=SYSTEM_SCALE),
        tooltip=[alt.Tooltip("system:N", title="System"), alt.Tooltip(f"{x}:N", title=x_title),
                 alt.Tooltip(f"{metric}:Q", format=".2%")] + ([alt.Tooltip("n:Q", title="Utterances")] if "n" in df else []),
    )


# ── API ─────────────────────────────────────────────────────────────────────────
def api_health():
    try:
        return requests.get(f"{API}/health", timeout=3).json()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def paper_results():
    r = requests.get(f"{API}/paper-results", timeout=30)
    r.raise_for_status()
    return r.json()


def post(endpoint, files, data):
    r = requests.post(f"{API}/{endpoint}", files=files, data=data, timeout=1800)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text[:500]}")
        return None
    return r.json()


# ── sidebar ─────────────────────────────────────────────────────────────────────
health = api_health()
with st.sidebar:
    st.header("Settings")
    if health is None:
        st.error(f"API not reachable at {API}.\n\nStart it with:\n\n`uvicorn api.main:app --port 8000`")
    else:
        st.success("API connected · models loaded" if health["models_loaded"] else "API connected · loading models")
        if not health["llm_available"]:
            st.warning(f"LLM stage unavailable: {health['llm_error']}")
    default_tau = health["default_tau"] if health else 0.30
    tau = st.slider("Pause threshold τ (s)", 0.05, 1.50, float(default_tau), 0.05,
                    help="The paper uses τ = 0.30 s, selected by the lift rule and validated on held-out speakers.")
    if abs(tau - default_tau) > 1e-9:
        st.caption(f"⚠️ Exploratory: the paper's results use τ = {default_tau:.2f} s.")
    run_baseline = st.checkbox("Run zero-shot Whisper-tiny", True)
    run_llm = st.checkbox("Run pause-guided LLM correction", True,
                          help="Only utterances with an inter-word gap ≥ τ are sent to the LLM (gpt-oss-120b via Groq).")
    st.caption("Pipeline: Whisper-tiny (encoder fine-tuned on TORGO) → CTC word alignment (wav2vec 2.0) → "
               "pause markers → LLM on ±4-word regions, behind an edit-distance guard.")

st.title("Pauses as Error Cues in Dysarthric ASR")
st.caption("Compare zero-shot Whisper-tiny, the TORGO fine-tuned model and pause-guided LLM post-correction.")

tab_try, tab_eval, tab_paper = st.tabs(["Try it", "Evaluate a set", "Paper results"])


# ── shared result views ────────────────────────────────────────────────────────
OP_STYLE = {"substitute": ("S", P["bad"]), "delete": ("D", P["bad"]), "insert": ("I", "#eda100" if THEME == "light" else "#c98500")}


def diff_html(ops):
    parts = []
    for o in ops:
        if o["op"] == "equal":
            parts.append(html.escape(o["hyp"]))
            continue
        tag, color = OP_STYLE[o["op"]]
        if o["op"] == "substitute":
            body = f'<s style="opacity:.6">{html.escape(o["ref"])}</s> {html.escape(o["hyp"])}'
        elif o["op"] == "delete":
            body = f'<s style="opacity:.6">{html.escape(o["ref"])}</s>'
        else:
            body = html.escape(o["hyp"])
        parts.append(f'<span title="{o["op"]}" style="border-bottom:2px solid {color};padding:0 2px">'
                     f'{body}<sup style="color:{color};font-weight:600">{tag}</sup></span>')
    return '<div style="line-height:2;font-size:15px">' + " ".join(parts) + "</div>"


def system_cards(result):
    systems = result["systems"]
    cols = st.columns(len(systems))
    ft = systems.get("finetuned", {}).get("metrics")
    for col, (key, s) in zip(cols, systems.items()):
        with col:
            st.markdown(f"**{s['label']}**")
            st.markdown(f"> {html.escape(s['text']) or '*(empty)*'}")
            m = s.get("metrics")
            if m:
                delta = (m["WER"] - ft["WER"]) if (ft and key != "finetuned") else None
                if delta is not None and abs(delta) < 0.0005:
                    st.metric("WER", f"{m['WER']:.1%}", "same as fine-tuned", delta_color="off", delta_arrow="off")
                else:
                    st.metric("WER", f"{m['WER']:.1%}", None if delta is None else f"{100 * delta:+.1f} pts vs fine-tuned",
                              delta_color="inverse")
                st.caption(f"CER {m['CER']:.1%} · MER {m['MER']:.1%} · S {m['S']} · D {m['D']} · I {m['I']} · "
                           f"{m['ref_words']} reference words")
                st.markdown(diff_html(m["alignment"]), unsafe_allow_html=True)


def metrics_chart(result):
    rows = [{"system": s["label"], "metric": k, "value": s["metrics"][k]}
            for s in result["systems"].values() if "metrics" in s for k in ("WER", "CER", "MER")]
    if not rows:
        return
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(**bar_kwargs()).encode(
        x=alt.X("metric:N", title=None, sort=["WER", "CER", "MER"], scale=GROUP_X, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("system:N", sort=SYSTEMS),
        y=alt.Y("value:Q", axis=pct_axis("Error rate")),
        color=alt.Color("system:N", scale=SYSTEM_SCALE),
        tooltip=["system:N", "metric:N", alt.Tooltip("value:Q", format=".2%")])
    show(chart, 260)
    table_view(df.pivot(index="system", columns="metric", values="value").reset_index())


def word_timeline(result):
    words = result.get("words")
    if not words:
        st.info("CTC alignment failed for this recording, so it has no word timings and no pause markers.")
        return
    # rows on a fixed 0–5 scale: gap labels (0.6), word bars (1.6–2.4), word labels alternating (3.0 / 4.0)
    w = pd.DataFrame(words)
    w["mid"] = (w.start + w.end) / 2
    w["label_y"] = [3.0 if i % 2 == 0 else 4.0 for i in range(len(w))]
    w["y0"], w["y1"] = 1.6, 2.4
    gaps = pd.DataFrame([{"start": a["end"], "end": a["end"] + (a["pause_after"] or 0), "gap": a["pause_after"] or 0,
                          "after": a["word"]} for a in words[:-1]])
    gaps = gaps[gaps.gap >= result["tau"]] if len(gaps) else gaps
    rows = alt.Scale(domain=[0, 5], nice=False, zero=False)
    y = alt.Y("y0:Q", axis=None, scale=rows)
    x = alt.X("start:Q", title="Time (s)", scale=alt.Scale(domain=[0, result["duration_s"]], nice=False))
    layers = []
    if len(gaps):
        gaps = gaps.assign(mid=(gaps.start + gaps.end) / 2, y0=1.2, y1=2.8, ly=0.6,
                           label=[f"{g:.2f} s" for g in gaps.gap])
        layers += [
            alt.Chart(gaps).mark_rect(color=P["series"][0], opacity=0.3).encode(
                x=x, x2="end:Q", y=y, y2="y1:Q", tooltip=[alt.Tooltip("gap:Q", title="pause (s)", format=".2f"), "after:N"]),
            alt.Chart(gaps).mark_text(fontSize=11, color=P["text"]).encode(
                x="mid:Q", y=alt.Y("ly:Q", axis=None, scale=rows), text="label:N"),
        ]
    layers += [
        alt.Chart(w).mark_rect(cornerRadius=4, color=P["series"][1], stroke=P["surface"], strokeWidth=2).encode(
            x=x, x2="end:Q", y=y, y2="y1:Q",
            tooltip=["word:N", alt.Tooltip("start:Q", format=".2f"), alt.Tooltip("end:Q", format=".2f"),
                     alt.Tooltip("pause_after:Q", title="gap after (s)", format=".2f"), alt.Tooltip("confidence:Q", format=".2f")]),
        alt.Chart(w).mark_text(fontSize=11, color=P["text2"]).encode(
            x="mid:Q", y=alt.Y("label_y:Q", axis=None, scale=rows), text="word:N"),
    ]
    show(alt.layer(*layers), 130)
    st.caption(f"Orange: words of the fine-tuned hypothesis (CTC timings). Blue: inter-word gaps ≥ τ = {result['tau']:.2f} s, "
               "which become pause markers.")
    table_view(w[["word", "start", "end", "duration", "pause_after", "confidence"]], "Word timings")


def llm_details(result):
    st.markdown("**Pause-formatted transcript**")
    st.code(result["formatted_text"] or "(empty)", language=None)
    if result.get("llm_note"):
        st.info(result["llm_note"])
    if not result["has_pause"]:
        st.info(f"No inter-word gap ≥ {result['tau']:.2f} s, so this recording is not sent to the LLM; "
                "the pause-guided output equals the fine-tuned transcript.")
    elif result["regions"]:
        st.markdown("**LLM regions** (±4 words around each marker; edits > 2 words or > 40% of a region are rejected)")
        st.dataframe(pd.DataFrame(result["regions"])[["input_region", "raw_llm_output", "validation_reason", "accepted_output"]],
                     width="stretch", hide_index=True)


# ── Try it ──────────────────────────────────────────────────────────────────────
with tab_try:
    left, right = st.columns([2, 3])
    with left:
        source = st.radio("Audio", ["Upload a file", "Record"], horizontal=True, label_visibility="collapsed")
        audio = (st.file_uploader("Audio file (WAV, FLAC or OGG; 16 kHz is ideal, others are resampled)", type=["wav", "flac", "ogg"])
                 if source == "Upload a file" else st.audio_input("Record a sentence"))
    with right:
        reference = st.text_area("Reference transcript (optional, enables WER/CER/MER)", height=100,
                                 placeholder="e.g. The quick brown fox jumps over the lazy dog")
        go = st.button("Transcribe and compare", type="primary", disabled=audio is None or health is None)
    if go:
        data = audio.getvalue()
        with st.spinner("Running Whisper, CTC alignment and the LLM stage…"):
            res = post("analyze", {"file": (getattr(audio, "name", None) or "recording.wav", data)},
                       {"reference": reference, "tau": tau, "run_baseline": run_baseline, "run_llm": run_llm})
        if res:
            st.session_state["single"] = (res, data)
    if "single" in st.session_state:
        res, data = st.session_state["single"]
        st.audio(data)
        st.caption(f"{res['filename']} · {res['duration_s']:.1f} s · τ = {res['tau']:.2f} s · "
                   + ("sent to the LLM" if res["sent_to_llm"] else "not sent to the LLM"))
        st.subheader("Transcripts")
        system_cards(res)
        if any("metrics" in s for s in res["systems"].values()):
            st.caption("Diff against the reference: S substitution (reference struck through), D deletion, I insertion. "
                       "WER delta is relative to the fine-tuned model.")
            st.subheader("Error rates")
            metrics_chart(res)
        st.subheader("Word timeline and pauses")
        word_timeline(res)
        st.subheader("LLM post-correction")
        llm_details(res)


# ── Evaluate a set ──────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("Upload several recordings, type or paste their reference transcripts, and compare the systems "
                "with corpus-level metrics (the same scoring as the paper).")
    files = st.file_uploader("Audio files", type=["wav", "flac", "ogg"], accept_multiple_files=True)
    refs_csv = st.file_uploader("Optional: CSV with columns filename,reference", type=["csv"])
    if files:
        known = {}
        if refs_csv is not None:
            known = dict(pd.read_csv(refs_csv).astype(str).values[:, :2])
        refs = st.data_editor(pd.DataFrame({"filename": [f.name for f in files],
                                            "reference": [known.get(f.name, "") for f in files]}),
                              disabled=["filename"], width="stretch", hide_index=True)
        if st.button("Evaluate", type="primary", disabled=health is None):
            with st.spinner(f"Processing {len(files)} recordings…"):
                res = post("evaluate", [("files", (f.name, f.getvalue())) for f in files],
                           {"references": json.dumps(dict(zip(refs.filename, refs.reference))),
                            "tau": tau, "run_baseline": run_baseline, "run_llm": run_llm})
            if res:
                st.session_state["batch"] = res
    if "batch" in st.session_state:
        res = st.session_state["batch"]
        summ = res["summary"]
        st.caption(f"{summ['files']} recordings, {summ['scored_files']} with a reference.")
        if summ["corpus"]:
            st.subheader("Corpus error rates")
            df = pd.DataFrame([{"system": c["label"], "metric": k, "value": c[k]}
                               for c in summ["corpus"].values() for k in ("WER", "CER", "MER")])
            show(alt.Chart(df).mark_bar(**bar_kwargs()).encode(
                x=alt.X("metric:N", title=None, sort=["WER", "CER", "MER"], scale=GROUP_X, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("system:N", sort=SYSTEMS), y=alt.Y("value:Q", axis=pct_axis("Error rate")),
                color=alt.Color("system:N", scale=SYSTEM_SCALE),
                tooltip=["system:N", "metric:N", alt.Tooltip("value:Q", format=".2%")]), 280)
            table_view(pd.DataFrame(summ["corpus"].values()))
        if summ["llm"]:
            l = summ["llm"]
            st.subheader("Effect of the LLM stage")
            a, b, c, d = st.columns(4)
            a.metric("Sent to the LLM", f"{l['sent_to_llm']} / {l['files']}")
            change = l["errors_after"] - l["errors_before"]
            if change:
                b.metric("Word errors after LLM", f"{l['errors_after']}", f"{change:+d} vs fine-tuned", delta_color="inverse")
            else:
                b.metric("Word errors after LLM", f"{l['errors_after']}", "same as fine-tuned", delta_color="off", delta_arrow="off")
            c.metric("Improved / worsened", f"{l['improved']} / {l['worsened']}")
            d.metric("Sign test p", "–" if l["sign_test_p"] is None else f"{l['sign_test_p']:.2f}")
        st.subheader("Per recording")
        rows = []
        for r in res["results"]:
            row = {"file": r["filename"], "pause ≥ τ": r["has_pause"]}
            for k, s in r["systems"].items():
                row[SYSTEM_KEYS[k]] = s["text"]
                if "metrics" in s:
                    row[f"WER · {SYSTEM_KEYS[k]}"] = round(100 * s["metrics"]["WER"], 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        pick = st.selectbox("Inspect a recording", [r["filename"] for r in res["results"]])
        r = next(x for x in res["results"] if x["filename"] == pick)
        system_cards(r)
        word_timeline(r)
        llm_details(r)


# ── Paper results ───────────────────────────────────────────────────────────────
with tab_paper:
    if health is None:
        st.info("Start the API to load the paper's results.")
        st.stop()
    try:
        R = paper_results()
    except Exception as e:
        st.error(f"Could not load the paper's results: {e}")
        st.stop()
    frame = lambda k: pd.DataFrame(R[k])
    asr, llm3, sel = frame("asr_test_split"), frame("llm_correction"), R["threshold_selection"]
    get = lambda s, g, m: asr[(asr.system == s) & (asr.speakers == g)][m].iloc[0]
    st.caption("TORGO, held-out test split (1,655 utterances) for ASR; all 16,552 utterances for the pause analysis "
               "and the LLM stage. Same run as the paper (τ = 0.30 s).")

    a, b, c, d = st.columns(4)
    a.metric("Fine-tuned test WER", f"{get(SYSTEMS[1], 'all', 'WER'):.2%}",
             f"zero-shot {get(SYSTEMS[0], 'all', 'WER'):.0%}", delta_color="off", delta_arrow="off")
    b.metric("Dysarthric test WER", f"{get(SYSTEMS[1], 'dysarthric', 'WER'):.2%}",
             f"healthy {get(SYSTEMS[1], 'healthy', 'WER'):.2%}", delta_color="off", delta_arrow="off")
    allr = llm3[llm3.set == "pause: all"].iloc[0]
    c.metric("LLM: word errors", f"{int(allr.errors_after) - int(allr.errors_before):+d}",
             f"{int(allr.errors_before):,} → {int(allr.errors_after):,} (363 utt.)", delta_color="off", delta_arrow="off")
    d.metric("Improved / worsened", f"{int(allr.improved)} / {int(allr.worsened)}", f"sign test p = {allr.sign_test_p:.2f}",
             delta_color="off", delta_arrow="off")

    st.subheader("ASR on the held-out test split (Table 2)")
    c1, c2 = st.columns([1, 3])
    metric = c1.radio("Metric", ["WER", "CER", "MER"], horizontal=True)
    with_zero = c1.checkbox("Include zero-shot", False, help="Zero-shot WER reaches 385% on dysarthric speech "
                                                             "(repeated-word insertions), which flattens the other bars.")
    d = asr if with_zero else asr[asr.system != SYSTEMS[0]]
    with c2:
        show(metric_bars(d, "speakers", metric, "Speakers", ["all", "dysarthric", "healthy"]), 280)
    table_view(asr)

    st.subheader("Per speaker, test split")
    sp = frame("asr_test_per_speaker")
    sp = sp if with_zero else sp[sp.system != SYSTEMS[0]]
    show(metric_bars(sp, "speaker", metric, "Speaker (F/M = dysarthric, FC/MC = healthy)",
                     sorted(sp.speaker.unique(), key=lambda s: (len(s), s))), 300)
    table_view(sp)

    st.subheader("Pause-threshold selection (Table 1, Fig. 2)")
    curve = frame("threshold_selection_curve")
    tau_sel = sel["tau_selected"]
    band = alt.Chart(curve).mark_area(opacity=0.15, color=P["series"][0]).encode(
        x=alt.X("tau:Q", title="Pause threshold τ (s)"), y=alt.Y("lift_ci_low:Q", axis=pct_axis("Lift over random positions")),
        y2="lift_ci_high:Q")
    line = alt.Chart(curve).mark_line(strokeWidth=2, color=P["series"][0], point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x="tau:Q", y="lift:Q",
        tooltip=[alt.Tooltip("tau:Q", title="τ (s)"), alt.Tooltip("lift:Q", format=".2%"), alt.Tooltip("lift_ci_low:Q", title="CI low", format=".2%"),
                 alt.Tooltip("lift_ci_high:Q", title="CI high", format=".2%"), "markers:Q", "speakers_with_min:Q", "supported:N"])
    unsupported = curve[~curve.supported]
    layers = [band, line, alt.Chart(pd.DataFrame({"tau": [tau_sel]})).mark_rule(color=P["text2"], strokeWidth=1.5).encode(x="tau:Q"),
              alt.Chart(pd.DataFrame({"tau": [tau_sel], "t": [f"selected τ = {tau_sel:.2f} s"]})).mark_text(
                  align="left", dx=6, dy=-6, color=P["text"], fontSize=12).encode(x="tau:Q", y=alt.value(12), text="t:N"),
              alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=P["axis"]).encode(y="y:Q")]
    if len(unsupported):
        layers.insert(0, alt.Chart(pd.DataFrame({"a": [unsupported.tau.min() - 0.025], "b": [curve.tau.max() + 0.025]}))
                      .mark_rect(color=P["neutral"], opacity=0.12).encode(x="a:Q", x2="b:Q"))
    show(alt.layer(*layers), 300)
    st.caption("Lift = reliability of pause markers minus that of random word boundaries in the same utterances "
               "(dysarthric speakers, 95% recording-level bootstrap CI). Grey: fewer than 7 of 8 speakers with ≥ 5 markers, "
               "so not eligible.")
    show(alt.Chart(curve).mark_line(strokeWidth=2, color=P["neutral"], point=alt.OverlayMarkDef(size=64, filled=True)).encode(
        x=alt.X("tau:Q", title="Pause threshold τ (s)"), y=alt.Y("markers:Q", title="Markers retained"),
        tooltip=["tau:Q", "markers:Q", "utterances:Q"]), 160)
    table_view(curve)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Leave-one-speaker-out validation**")
        loso = frame("threshold_selection_loso")
        show(alt.Chart(loso).mark_bar(**bar_kwargs(), color=P["series"][0]).encode(
            x=alt.X("held_out_speaker:N", title="Held-out speaker", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("heldout_lift:Q", axis=pct_axis("Lift on held-out speaker")),
            tooltip=["held_out_speaker:N", alt.Tooltip("tau_selected_on_others:Q", title="τ chosen on the other 7"),
                     "heldout_markers:Q", alt.Tooltip("heldout_lift:Q", format=".2%")]), 240)
        st.caption(f"Pooled over held-out speakers: lift {sel['loso_heldout_lift']:.2%} "
                   f"(permutation p = {sel['loso_perm_p']:.3f}). τ chosen on the other speakers: "
                   f"{loso.tau_selected_on_others.min():.2f}–{loso.tau_selected_on_others.max():.2f} s.")
        table_view(loso)
    with c2:
        st.markdown("**Robustness at the selected and manual τ**")
        st.dataframe(frame("threshold_selection_robustness"), width="stretch", hide_index=True)

    st.subheader("Pause markers vs random word boundaries")
    pv = frame("pause_vs_random")
    long = pd.concat([
        pv.assign(kind="Pause markers", reliability_v=pv.reliability, coverage_v=pv.error_coverage),
        pv.assign(kind="Random positions", reliability_v=pv.random_reliability, coverage_v=pv.random_error_coverage)])
    kind_scale = alt.Scale(domain=["Pause markers", "Random positions"], range=[P["series"][0], P["neutral"]])
    c1, c2 = st.columns(2)
    for col, field, title in [(c1, "reliability_v", "Reliability (marker within 2 words of an error)"),
                              (c2, "coverage_v", "Error coverage")]:
        with col:
            show(alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
                x=alt.X("threshold:Q", title="Pause threshold τ (s)"), y=alt.Y(f"{field}:Q", axis=pct_axis(title)),
                color=alt.Color("kind:N", scale=kind_scale),
                tooltip=["kind:N", "threshold:Q", "markers:Q", alt.Tooltip(f"{field}:Q", format=".2%"),
                         alt.Tooltip("p_value:Q", format=".4f")]), 260)
    st.caption("570 utterances with an inter-word gap ≥ 0.02 s (1,617 ASR errors). Pause markers sit near errors "
               "slightly more often than random positions, but cover about as many errors: a weak positional cue.")
    table_view(pv)

    st.subheader("Inter-word gap distribution")
    gd = frame("gap_distribution")
    gd = gd[gd.bin >= 0.05].copy()
    gd["share"] = gd.boundaries / gd.groupby("status").boundaries.transform("sum")
    gd["group"] = gd.status.map({"dysarthria": "Dysarthric", "healthy": "Healthy"})
    show(alt.layer(
        alt.Chart(gd).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
            x=alt.X("bin:Q", title="Gap between consecutive words (s, 0.05 s bins; last bin ≥ 2 s)"),
            y=alt.Y("share:Q", axis=pct_axis("Share of gaps ≥ 0.05 s")),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Dysarthric", "Healthy"], range=P["series"][:2])),
            tooltip=["group:N", "bin:Q", "boundaries:Q", alt.Tooltip("share:Q", format=".2%")]),
        alt.Chart(pd.DataFrame({"t": [tau_sel]})).mark_rule(color=P["text2"]).encode(x="t:Q")), 260)
    table_view(gd)

    st.subheader("Pause-guided LLM correction (Table 3)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What happened to the 370 regions**")
        ro = frame("llm_region_outcomes")
        show(alt.Chart(ro).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=P["series"][2]).encode(
            y=alt.Y("outcome:N", title=None, sort=["returned unchanged", "edited", "rejected by guard"]),
            x=alt.X("regions:Q", title="Regions"), tooltip=["outcome:N", "regions:Q"]) +
             alt.Chart(ro).mark_text(align="left", dx=4, color=P["text2"]).encode(
                 y=alt.Y("outcome:N", sort=["returned unchanged", "edited", "rejected by guard"]), x="regions:Q", text="regions:Q"), 160)
    with c2:
        st.markdown("**Utterances improved vs worsened, per speaker**")
        ps = frame("llm_per_speaker")
        div = pd.concat([ps.assign(effect="Improved", count=ps.improved), ps.assign(effect="Worsened", count=-ps.worsened)])
        div = div[div["count"] != 0]
        show(alt.Chart(div).mark_bar(cornerRadius=4, stroke=P["surface"], strokeWidth=2).encode(
            x=alt.X("speaker:N", title="Speaker", sort=sorted(ps.speaker, key=lambda s: (len(s), s)), axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Utterances (worsened below 0)"),
            color=alt.Color("effect:N", scale=alt.Scale(domain=["Improved", "Worsened"], range=[P["good"], P["bad"]])),
            tooltip=["speaker:N", "effect:N", alt.Tooltip("count:Q", title="utterances"), "utterances:Q",
                     "errors_before:Q", "errors_after:Q"]), 220)
        table_view(ps)
    st.dataframe(llm3, width="stretch", hide_index=True)
    ex = frame("llm_changed_examples")
    outcome = st.segmented_control("Changed utterances", ["all", "improved", "worsened", "unchanged"], default="all")
    st.dataframe(ex if outcome in (None, "all") else ex[ex.outcome == outcome], width="stretch", hide_index=True)

    st.subheader("Other checks")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**WER by split** (train = test, so no memorisation)")
        st.dataframe(frame("wer_by_split"), width="stretch", hide_index=True)
    with c2:
        st.markdown("**Utterance-level association** (do pauses flag erroneous utterances?)")
        st.dataframe(frame("utterance_level_association"), width="stretch", hide_index=True)
