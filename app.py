# --- Ensure package imports work in every environment (incl. Streamlit Cloud) ---
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import re, io, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shared.parsing import parse_financials, compute_ratios, benchmark_for, BENCHMARKS, default_keywords
from parser_pdf import extract_tables_to_long
from export_docx import memo_to_docx

st.set_page_config(page_title="ClearPass — Underwriting", layout="wide")
st.title("🧮 ClearPass — Underwriting & Financial Health")

left, right = st.columns([3,2], gap="large")

def ai_like_summary(ratios):
    def strength(val, good, ok, inverse=False):
        if val is None or (isinstance(val,float) and np.isnan(val)): return 'n/a'
        if inverse:
            if val <= good: return 'strong'
            if val <= ok: return 'acceptable'
            return 'elevated'
        else:
            if val >= good: return 'strong'
            if val >= ok: return 'acceptable'
            return 'weak'
    cr = ratios.get('Current Ratio'); qr = ratios.get('Quick Ratio')
    de = ratios.get('Debt-to-Equity'); pm = ratios.get('Profit Margin (%)'); roa = ratios.get('Return on Assets (%)')
    cov = ratios.get('Interest Coverage (EBIT)') or ratios.get('Interest Coverage (EBITDA)')
    dscr = ratios.get('DSCR (CFO / Debt Service)')
    lines = []
    lines.append(f"Liquidity — Current {cr}, Quick {qr}. Position appears {strength(cr,1.8,1.2)}.")
    lines.append(f"Leverage — D/E {de}. Leverage is {strength(de,0.8,1.5,inverse=True)}.")
    lines.append(f"Profitability — Margin {pm}% and ROA {roa}%. Profitability is {strength(pm,12,6)} relative to medians.")
    if cov is not None: lines.append(f"Coverage — Interest coverage ≈ {cov}x ({'adequate' if cov>=3 else 'tight'}).")
    if dscr is not None: lines.append(f"DSCR — {dscr}x; ≥1.25x preferred for term debt.")
    return "\n".join(lines)

def underwriting_memo(company, year, industry, basics, ratios, bench):
    def fmt(v, pct=False):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "n/a"
        return f"{v:,.2f}%" if pct else f"{v:,.0f}"
    def rfmt(v): 
        if v is None or (isinstance(v,float) and np.isnan(v)): return "n/a"
        return f"{v:,.2f}"
    lines = []
    lines += [f"Underwriting Memo — {company} (FY {year})", f"Industry: {industry}", "—"*60]
    lines += ["Executive Summary"]
    lines += [
        f"Liquidity: Current {rfmt(ratios.get('Current Ratio'))} (bench {bench['Current Ratio']}), "
        f"Quick {rfmt(ratios.get('Quick Ratio'))} (bench {bench['Quick Ratio']}).",
        f"Leverage: D/E {rfmt(ratios.get('Debt-to-Equity'))} (bench {bench['Debt-to-Equity']}).",
        f"Profitability: Margin {rfmt(ratios.get('Profit Margin (%)'))}% (bench {bench['Profit Margin (%)']}%), "
        f"ROA {rfmt(ratios.get('Return on Assets (%)'))}% (bench {bench['Return on Assets (%)']}%).",
        f"Coverage: Interest coverage ≈ {rfmt(ratios.get('Interest Coverage (EBIT)') or ratios.get('Interest Coverage (EBITDA)'))}x "
        f"(target ≥3x). DSCR {rfmt(ratios.get('DSCR (CFO / Debt Service)'))}x (preferred ≥1.25x)."
    ]
    lines += ["", "Financial Snapshot"]
    for k in ["Revenue","COGS","Operating Expenses","EBIT","EBITDA","Net Income","Cash","Accounts Receivable","Inventory",
              "Current Assets","Current Liabilities","Total Liabilities","Equity","Total Assets","CFO","Interest Expense","Interest Paid","Principal Repayment"]:
        lines.append(f"{k}: {fmt(basics.get(k), pct=False)}")
    lines += ["", "Key Risks",
              "- Working capital strain if AR extends or inventory turns slow.",
              "- Margin pressure in downturn or input cost shock.",
              "- Exposure to rising rates on floating debt.",
              "", "Mitigants",
              "- Positive CFO / acceptable coverage.",
              "- Cost flexibility in SG&A.",
              "- Leverage within/near sector medians.",
              "", "Indicative Decision Framework",
              "• Approve if: D/E ≤ 1.5x, Interest coverage ≥ 3x, Current ratio ≥ 1.2x, DSCR ≥ 1.25x.",
              "• Approve with conditions/LOC if marginal on one dimension.",
              "• Decline/collateralize if: coverage <2x or DSCR <1.0x."]
    return "\n".join(lines)

with left:
    st.subheader("Upload (CSV/XLSX/PDF)")
    files = st.file_uploader("Upload one or more statements. We'll merge and parse automatically.", 
                              type=["csv","xlsx","pdf"], accept_multiple_files=True)
    company = st.text_input("Company Name", "DemoCo Ltd.")
    fiscal_year = st.text_input("Fiscal Year", "2024")
    industry = st.selectbox("Industry", BENCHMARKS["industry_name"].tolist(), index=1)

    dfs = []
    wide_candidate = None
    if files:
        for f in files:
            try:
                if f.name.lower().endswith(".pdf"):
                    df = extract_tables_to_long(f)
                elif f.name.lower().endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f, sheet_name=0)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue
            if df.shape[1] >= 3:
                wide_candidate = df.copy()
            dfs.append(df)
    if not dfs:
        sample = pd.DataFrame({
            "Line Item":["Revenue","COGS","Operating Expenses","EBIT","Net Income","Cash","Accounts Receivable","Inventory",
                         "Current Assets","Current Liabilities","Total Liabilities","Equity","Total Assets","Interest Expense",
                         "Net cash provided by operating activities","Repayments of borrowings"],
            "2022":[1_000_000,600_000,250_000,150_000,90_000,50_000,40_000,30_000,150_000,80_000,220_000,300_000,520_000,20_000,85_000,15_000],
            "2023":[1_200_000,720_000,300_000,180_000,120_000,60_000,45_000,28_000,160_000,81_000,230_000,320_000,550_000,22_000,95_000,18_000],
            "2024":[1_350_000,800_000,335_000,215_000,150_000,62_000,50_000,25_000,170_000,85_000,240_000,340_000,580_000,24_000,110_000,20_000]
        })
        dfs = [sample]
        wide_candidate = sample

    merged = pd.concat(dfs, ignore_index=True)
    st.markdown("**Preview**")
    st.dataframe(merged.head(25))

    basics, _ = parse_financials(merged, default_keywords())
    ratios = compute_ratios(basics)
    bench = benchmark_for(industry)

with right:
    st.subheader("Key Ratios")
    for key in ['Current Ratio','Quick Ratio','Debt-to-Equity','Profit Margin (%)','Return on Assets (%)','Interest Coverage (EBIT)','DSCR (CFO / Debt Service)']:
        st.metric(key, 'n/a' if (ratios.get(key) is None or (isinstance(ratios.get(key), float) and np.isnan(ratios.get(key)))) else ratios[key])
    st.subheader("AI-Style Summary")
    st.markdown(ai_like_summary(ratios))

st.divider()
st.subheader("Trend Charts (if multi-year provided)")
if wide_candidate is not None:
    years = [c for c in wide_candidate.columns if re.search(r"20\d\d", str(c))]
    if len(years) >= 2:
        rows = []
        for y in years:
            dfy = wide_candidate[[wide_candidate.columns[0], y]].rename(columns={wide_candidate.columns[0]: "Account", y: "Value"})
            b, _ = parse_financials(dfy, default_keywords())
            r = compute_ratios(b)
            r["Year"] = y
            rows.append(r)
        dfy = pd.DataFrame(rows)
        for metric in ["Current Ratio","Debt-to-Equity","Profit Margin (%)"]:
            fig, ax = plt.subplots()
            ax.plot(dfy["Year"], dfy[metric], marker="o")
            ax.set_title(metric)
            st.pyplot(fig)

st.divider()
st.subheader("Exports")
c1, c2 = st.columns(2)
with c1:
    if st.button("Generate Underwriting PDF"):
        from matplotlib.backends.backend_pdf import PdfPages
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0,0,1,1]); ax.axis("off")
            ax.add_patch(plt.Rectangle((0,0.93), 1,0.07, transform=ax.transAxes))
            ax.text(0.06, 0.965, 'ClearPass — Underwriting Report', fontsize=18, weight='bold', transform=ax.transAxes, va='center')
            ax.text(0.06, 0.84, company, fontsize=22, weight='bold')
            ax.text(0.06, 0.80, f'Fiscal Year: {fiscal_year}', fontsize=11)
            ax.text(0.06, 0.77, f'Industry: {industry}', fontsize=11)
            pdf.savefig(fig); plt.close(fig)

            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.08,0.12,0.84,0.78]); ax.axis("off")
            ax.set_title("Key Ratios & Benchmarks", loc="left", fontsize=16, pad=10)
            rows = [
                ("Current Ratio", ratios.get("Current Ratio"), bench["Current Ratio"]),
                ("Quick Ratio", ratios.get("Quick Ratio"), bench["Quick Ratio"]),
                ("Debt-to-Equity", ratios.get("Debt-to-Equity"), bench["Debt-to-Equity"]),
                ("Profit Margin (%)", ratios.get("Profit Margin (%)"), bench["Profit Margin (%)"]),
                ("Return on Assets (%)", ratios.get("Return on Assets (%)"), bench["Return on Assets (%)"]),
                ("Interest Coverage (EBIT)", ratios.get("Interest Coverage (EBIT)"), "≥3.0x target"),
                ("Interest Coverage (EBITDA)", ratios.get("Interest Coverage (EBITDA)"), "≥3.0x target"),
                ("DSCR (CFO / Debt Service)", ratios.get("DSCR (CFO / Debt Service)"), "≥1.25x preferred"),
            ]
            y=0.95
            for name, val, b in rows:
                ax.text(0.02,y,f"{name}", fontsize=11)
                ax.text(0.55,y,f"{'n/a' if (val is None or (isinstance(val,float) and np.isnan(val))) else round(val,2)}", fontsize=11)
                ax.text(0.78,y,f"{'' if (b is None) else b}", fontsize=11)
                y -= 0.06
            pdf.savefig(fig); plt.close(fig)

            memo = underwriting_memo(company, fiscal_year, industry, basics, ratios, bench)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.08,0.08,0.84,0.84]); ax.axis("off")
            ax.set_title("Underwriting Memo", loc="left", fontsize=16, pad=10)
            wrapped = textwrap.fill(memo, 110)
            ax.text(0,1, wrapped, va="top", fontsize=10)
            pdf.savefig(fig); plt.close(fig)
        buf.seek(0)
        st.download_button("⬇️ Download PDF", data=buf, file_name=f"{company}_Underwriting_Report.pdf")

with c2:
    if st.button("Download DOCX Memo"):
        memo = underwriting_memo(company, fiscal_year, industry, basics, ratios, bench)
        out = io.BytesIO()
        tmp_path = f"/tmp/{company}_Underwriting_Memo.docx"
        memo_to_docx(memo, tmp_path)
        with open(tmp_path, "rb") as fh:
            out.write(fh.read())
        out.seek(0)
        st.download_button("⬇️ Download DOCX", data=out, file_name=f"{company}_Underwriting_Memo.docx")

st.divider()
st.subheader("Parsed Basics (latest year)")
st.json({k:(None if (v is None or (isinstance(v,float) and np.isnan(v))) else float(v)) for k,v in basics.items()})
st.subheader("All Ratios")
st.json(ratios)
