import pandas as pd
from shared.parsing import parse_financials, compute_ratios

def test_parse_long_minimal():
    df = pd.read_csv('samples/sample_public_company_long.csv')
    basics, _ = parse_financials(df)
    ratios = compute_ratios(basics)
    assert round(basics['Revenue'],2) == 1350000.0
    assert round(ratios['Profit Margin (%)'],2) == round(150000/1350000*100,2)
    assert round(ratios['Current Ratio'],2) == round(170000/85000,2)
