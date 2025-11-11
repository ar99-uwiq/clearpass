import re
import pandas as pd
import numpy as np
from typing import Dict

BENCHMARKS = pd.DataFrame([
    (311,'Food Manufacturing',1.5,1.2,1.2,8.0,6.0),
    (423,'Wholesale Trade',1.6,1.3,1.0,6.0,5.0),
    (424,'Merchant Wholesalers',1.6,1.3,1.0,6.0,5.0),
    (44,'Retail',1.4,1.1,1.6,4.0,4.0),
    (48,'Transportation/Logistics',1.3,1.0,2.0,3.0,3.0),
    (51,'Information/Software',2.0,1.8,0.6,12.0,10.0),
    (52,'Financial Services',1.5,1.3,1.5,10.0,8.0),
    (54,'Professional Services',1.8,1.6,0.8,12.0,10.0),
    (31,'Manufacturing (General)',1.5,1.2,1.2,8.0,6.0),
], columns=['naics','industry_name','current_ratio_median','quick_ratio_median','d_to_e_median','profit_margin_median','roa_median'])

def default_keywords():
    return {
        'current_assets':[r'\bcurrent assets\b', r'\btotal current assets\b'],
        'cash':[r'\bcash\b', r'\bcash and cash equivalents\b', r'\bcash equivalents\b'],
        'accounts_receivable':[r'\baccounts receivable\b', r'\btrade receivables\b', r'\breceivables\b'],
        'inventory':[r'\binventor(y|ies)\b', r'\bmerchandise inventory\b', r'\bstock[- ]in[- ]trade\b'],
        'current_liabilities':[r'\bcurrent liabilities\b', r'\btotal current liabilities\b'],
        'total_liabilities':[r'\btotal liabilities\b', r'\bliabilities\b(?!.*and equity)'],
        'equity':[r"\b(total )?(shareholders'|stockholders'|owners'?) equity\b", r'\btotal equity\b', r'\bequity attributable\b'],
        'total_assets':[r'\btotal assets\b'],
        'revenue':[r'\b(revenue|sales|net sales|total revenue)\b', r'\bturnover\b'],
        'cogs':[r'\b(cost of goods sold|cogs|cost of revenue)\b'],
        'operating_expenses':[r'\boperating expenses\b', r'\bselling, general and administrative\b', r'\bsga\b', r'\bresearch and development\b'],
        'ebit':[r'\boperating income\b', r'\bebit\b', r'\bearnings before interest and taxes\b'],
        'ebitda':[r'\bebitda\b', r'\bearnings before interest, taxes, depreciation and amortization\b'],
        'interest_expense':[r'\binterest expense\b', r'\bfinance costs?\b'],
        'net_income':[r'\bnet income\b', r'\bprofit attributable\b', r'\bnet profit\b', r'\bprofit for the period\b'],
        'short_term_debt':[r'\bshort[- ]?term debt\b', r'\bcurrent portion of (long[- ]?term )?debt\b'],
        'long_term_debt':[r'\blong[- ]?term debt\b', r'\bnon[- ]?current borrowings\b'],
        'accounts_payable':[r'\baccounts payable\b', r'\btrade payables\b'],
        'cfo':[r'\bnet cash provided by operating activities\b', r'\bcash flow from operations\b'],
        'interest_paid':[r'\binterest paid\b'],
        'principal_repayment':[r'\b(principal|loan) repayments?\b', r'\brepayments? of borrowings?\b']
    }

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _parse_wide(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if re.search(r"(20\d\d)|(\bfy\d{2}\b)", str(c).lower())]
    if not cols:
        out = df.iloc[:, :2].copy()
        out.columns = ["Account","Value"]
        return out
    def year_key(c):
        m = re.search(r"(20\d\d)", str(c))
        return int(m.group(1)) if m else -1
    latest = sorted(cols, key=year_key)[-1]
    out = pd.DataFrame({"Account": df.iloc[:,0], "Value": df[latest]})
    return out

def parse_financials(input_df: pd.DataFrame, kw_map=None):
    if kw_map is None:
        kw_map = default_keywords()
    df = input_df.copy().dropna(how="all")
    if df.shape[1] >= 3:
        try:
            df = _parse_wide(df)
        except Exception:
            df = df.iloc[:, :2]
            df.columns = ["Account","Value"]
    else:
        df.columns = ["Account","Value"]
    df["Account"] = df["Account"].astype(str)
    df["Value"] = df["Value"].astype(str).str.replace(",","", regex=False).str.replace("(","-", regex=False).str.replace(")","", regex=False)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    agg = {k:0.0 for k in kw_map.keys()}
    for _, row in df.iterrows():
        name = _norm(row["Account"])
        for k, pats in kw_map.items():
            if any(re.search(p, name) for p in pats):
                if pd.notnull(row["Value"]):
                    agg[k] += float(row["Value"])

    current_assets = agg.get('current_assets') or ( (agg.get('cash') or 0)+(agg.get('accounts_receivable') or 0)+(agg.get('inventory') or 0) )
    basics = {
        'Revenue': agg.get('revenue') or np.nan,
        'COGS': agg.get('cogs') or np.nan,
        'Operating Expenses': agg.get('operating_expenses') or np.nan,
        'EBIT': agg.get('ebit') if agg.get('ebit') else ( (agg.get('revenue') or 0) - (agg.get('cogs') or 0) - (agg.get('operating_expenses') or 0) if all([(agg.get('revenue') or 0),(agg.get('cogs') or 0)]) else np.nan ),
        'EBITDA': agg.get('ebitda') or np.nan,
        'Net Income': agg.get('net_income') or np.nan,
        'Cash': agg.get('cash') or np.nan,
        'Accounts Receivable': agg.get('accounts_receivable') or np.nan,
        'Inventory': agg.get('inventory') or np.nan,
        'Accounts Payable': agg.get('accounts_payable') or np.nan,
        'Short-term Debt': agg.get('short_term_debt') or np.nan,
        'Long-term Debt': agg.get('long_term_debt') or np.nan,
        'Current Assets': current_assets if current_assets!=0 else np.nan,
        'Current Liabilities': agg.get('current_liabilities') or np.nan,
        'Total Liabilities': agg.get('total_liabilities') or np.nan,
        'Equity': agg.get('equity') or np.nan,
        'Total Assets': agg.get('total_assets') or np.nan,
        'Interest Expense': agg.get('interest_expense') or np.nan,
        'CFO': agg.get('cfo') or np.nan,
        'Interest Paid': agg.get('interest_paid') or np.nan,
        'Principal Repayment': agg.get('principal_repayment') or np.nan
    }
    return basics, agg

def _safe_div(a,b):
    if b in (None,0) or (isinstance(b,float) and pd.isna(b)): return np.nan
    if a is None or (isinstance(a,float) and pd.isna(a)): return np.nan
    return float(a)/float(b)

def compute_ratios(basics: Dict[str,float]):
    ca = basics.get('Current Assets'); cl = basics.get('Current Liabilities')
    cash = basics.get('Cash'); ar = basics.get('Accounts Receivable'); inv = basics.get('Inventory')
    tl = basics.get('Total Liabilities'); eq = basics.get('Equity'); ta = basics.get('Total Assets')
    rev = basics.get('Revenue'); ni = basics.get('Net Income')
    ebit = basics.get('EBIT'); ebitda = basics.get('EBITDA'); int_exp = basics.get('Interest Expense')
    quick_assets = ((cash or 0) + (ar or 0)) if any([cash, ar]) else ((ca or 0) - (inv or 0))
    ratios = {
        'Current Ratio': _safe_div(ca, cl),
        'Quick Ratio': _safe_div(quick_assets, cl),
        'Debt-to-Equity': _safe_div(tl, eq),
        'Profit Margin (%)': (_safe_div(ni, rev)*100) if rev not in (None,0) else np.nan,
        'Return on Assets (%)': (_safe_div(ni, ta)*100) if ta not in (None,0) else np.nan,
        'Interest Coverage (EBIT)': _safe_div(ebit, int_exp),
        'Interest Coverage (EBITDA)': _safe_div(ebitda, int_exp),
        'Gross Margin (%)': (_safe_div((rev - (basics.get('COGS') or 0)), rev)*100) if rev not in (None,0) else np.nan,
        'Operating Margin (%)': (_safe_div(ebit, rev)*100) if rev not in (None,0) else np.nan
    }
    cfo = basics.get('CFO'); principal = basics.get('Principal Repayment'); interest_paid = basics.get('Interest Paid') or int_exp
    denom = (interest_paid or 0) + (principal or 0)
    ratios['DSCR (CFO / Debt Service)'] = None if denom==0 else round((cfo or 0)/denom, 2)
    out = {}
    for k,v in ratios.items():
        if v is None or (isinstance(v,float) and pd.isna(v)): out[k] = None
        else: out[k] = round(float(v),2)
    return out

def benchmark_for(industry: str):
    row = BENCHMARKS[BENCHMARKS['industry_name']==industry]
    if row.empty: row = BENCHMARKS.iloc[[0]]
    r = row.iloc[0].to_dict()
    return {
        'Current Ratio': r['current_ratio_median'],
        'Quick Ratio': r['quick_ratio_median'],
        'Debt-to-Equity': r['d_to_e_median'],
        'Profit Margin (%)': r['profit_margin_median'],
        'Return on Assets (%)': r['roa_median'],
    }
