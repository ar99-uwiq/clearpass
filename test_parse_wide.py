import pandas as pd
from shared.parsing import parse_financials, compute_ratios

def test_parse_wide_latest_year():
    df = pd.read_csv('samples/sample_public_company_wide.csv')
    df2 = df[['Line Item', '2024']].rename(columns={'Line Item':'Account','2024':'Value'})
    basics, _ = parse_financials(df2)
    ratios = compute_ratios(basics)
    assert basics['Revenue'] == 1350000
    assert ratios['Current Ratio'] == round(170000/85000,2)
