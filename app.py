}

/* White ROC-AUC card: override the general white-text sidebar rule */
section[data-testid="stSidebar"] div[data-testid="stMetric"],
section[data-testid="stSidebar"] div[data-testid="stMetric"] *,
section[data-testid="stSidebar"] div[data-testid="stMetric"] label {
  color: #12263F !important;
  -webkit-text-fill-color: #12263F !important;
}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {
  background: #FFFFFF !important;
  border: 1px solid #DCE4EE !important;
  border-radius: 16px !important;
}

/* Help icons remain visible on the light form background */
section[data-testid="stMain"] [data-testid="stTooltipIcon"] {
  color: #53657B !important;
}

/* -------------------------------------------------------------------------
   Strong Streamlit widget contrast overrides.
   Streamlit/BaseWeb labels are not always nested under stWidgetLabel, so
   target the actual widget containers as well. Layout and widget positions
   are intentionally unchanged.
   ------------------------------------------------------------------------- */
section[data-testid="stMain"] label,
section[data-testid="stMain"] label *,
section[data-testid="stMain"] [data-testid="stWidgetLabel"],
section[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
section[data-testid="stMain"] .stSlider p,
section[data-testid="stMain"] .stSlider span,
section[data-testid="stMain"] [data-testid="stSlider"] p,
section[data-testid="stMain"] [data-testid="stSlider"] span {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
  opacity: 1 !important;
}

/* Supporting descriptions and slider endpoints */
section[data-testid="stMain"] [data-testid="stCaptionContainer"],
section[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
section[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"],
section[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] * {
  color: #53657B !important;
  -webkit-text-fill-color: #53657B !important;
  opacity: 1 !important;
}

/* Help / tooltip icons */
section[data-testid="stMain"] [data-testid="stTooltipIcon"],
section[data-testid="stMain"] [data-testid="stTooltipIcon"] *,
section[data-testid="stMain"] [data-testid="stTooltipIcon"] svg {
  color: #53657B !important;
  fill: #53657B !important;
  opacity: 1 !important;
}

/* Consistent light input surfaces and dark entered values */
section[data-testid="stMain"] [data-baseweb="input"],
section[data-testid="stMain"] [data-baseweb="base-input"],
section[data-testid="stMain"] [data-baseweb="select"] > div {
  background-color: #FFFFFF !important;
  border-color: #C9D5E3 !important;
}
section[data-testid="stMain"] input,
section[data-testid="stMain"] [data-baseweb="select"] input,
section[data-testid="stMain"] [data-baseweb="select"] div {
  color: #19324F !important;
  -webkit-text-fill-color: #19324F !important;
}

/* Main section headings stay dark and clear */
section[data-testid="stMain"] h1,
section[data-testid="stMain"] h2,
section[data-testid="stMain"] h3,
section[data-testid="stMain"] h4 {
  color: #12263F !important;
}

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render HTML without Markdown treating indented lines as code blocks."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize()


                                    help=f"Allowed range: {int(lo)} to {int(hi)}")
            else:
                st.number_input(label, min_value=lo, max_value=hi, key=key,
                                help=f"Allowed range: {lo:g} to {hi:g}")
        else:
            st.selectbox(label, f.get("options", ["Unknown"]), key=key)

    with st.form("passenger_form"):
        for title, note, group, ncols in (
            ("Passenger profile", "Who is travelling and how.", category_feats, 3),
            ("Flight details", "Distance, timing and other measurements.", number_feats, 3),
            ("Service ratings", "Rate each service from lowest to highest.", rating_feats, 2),
        ):
            if not group:
                continue
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(note)
                cols = st.columns(ncols)
                for i, f in enumerate(group):
                    with cols[i % ncols]:
                        render_feature(f)

        submitted = st.form_submit_button(
            "Predict passenger satisfaction", type="primary", use_container_width=True
        )

    if submitted:
        values = {f["name"]: st.session_state[key_of(f)] for f in features}
        input_df = pd.DataFrame([values])
        try:
            probability = float(model.predict_proba(input_df)[0, 1])
        except Exception as error:
            st.error(f"Prediction failed: {error}. Check that the inputs match the trained model's schema.")
            st.stop()

        satisfied = probability >= 0.5
        cls = "ok" if satisfied else "no"
        verdict = "Satisfied" if satisfied else "Neutral or dissatisfied"
        confidence = max(probability, 1 - probability)
        if confidence >= 0.85:
            strength = "High confidence"
        elif confidence >= 0.65:
            strength = "Moderate confidence"
        else:
            strength = "Close call"
        detail = (
            "This passenger is likely to be happy with the journey."
            if satisfied else
            "This passenger is unlikely to be happy. See the suggestions below."
        )

        html(f"""
        <div class="pass">
        <div class="stub {cls}">
        <div class="label">Probability of satisfaction</div>
        <div class="big">{probability:.1%}</div>
        <div class="label">{strength}</div>
        </div>
        <div class="perf"></div>
        <div class="body">
        <div class="verdict">{verdict}</div>
        <div class="sub">{detail}</div>
        <div class="meter"><div class="mask" style="left:{probability*100:.1f}%"></div><div class="tick"></div></div>
        <div class="meter-legend"><span>Unlikely</span><span>50% decision line</span><span>Very likely</span></div>
        </div>
        </div>
        """)

        # What-if insights: raise each rating to its best score, one at a time.
        gains = []
        candidates, meta = [], []
        for f in rating_feats:
            best = int(float(f["max"]))
            if values[f["name"]] < best:
                row = dict(values)
                row[f["name"]] = best
                candidates.append(row)
                meta.append((pretty(f["name"]), values[f["name"]], best))
        if candidates:
            try:
                new_probs = model.predict_proba(pd.DataFrame(candidates))[:, 1]
                gains = sorted(
                    ((n, cur, best, float(p) - probability) for (n, cur, best), p in zip(meta, new_probs)),
                    key=lambda x: x[3], reverse=True,
                )
            except Exception:
                gains = []

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**What-if insights**")
            top = [g for g in gains if g[3] > 0.001][:3]
            if top:
                st.caption("Estimated change if a single rating were raised to its best score.")
                for n, cur, best, delta in top:
                    with st.container(border=True):
                        c_a, c_b = st.columns([3, 1])
                        c_a.markdown(f"**{n}**  \nfrom {cur} to {best}")
                        c_b.metric("Gain", f"+{delta*100:.1f} pts", label_visibility="collapsed")
            elif candidates:
                st.info("Improving any single service rating would not noticeably change this result.")
            else:
                st.info("All service ratings are already at their best score.")
        with right:
            st.markdown("**Save this result**")
            st.caption("Download the inputs and prediction as a CSV file.")
            out = input_df.copy()
            out["satisfaction_probability"] = round(probability, 4)
            out["prediction"] = verdict
            st.download_button("Download result (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="satisfaction_prediction.csv", mime="text/csv",
                               use_container_width=True)

        st.session_state["history"].append({
            "Run": len(st.session_state["history"]) + 1,
            "Probability": f"{probability:.1%}",
            "Prediction": verdict,
        })

    if st.session_state["history"]:
        with st.expander(f"Session history ({len(st.session_state['history'])} predictions)"):
            st.dataframe(pd.DataFrame(st.session_state["history"]).iloc[::-1],
                         use_container_width=True, hide_index=True)
            if st.button("Clear history"):
                st.session_state["history"] = []
                st.rerun()
