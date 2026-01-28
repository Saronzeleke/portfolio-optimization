def get_price_column(df):
    for c in df.columns:
        if 'Adj Close' in c:
            return c
    for c in df.columns:
        if 'Close' in c:
            return c
    raise ValueError("No price column found")
