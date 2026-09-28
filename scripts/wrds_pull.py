"""Pull the CRSP daily and monthly return panels from WRDS.

Run on a machine with WRDS access; the wrds package prompts for credentials.

    pip install wrds pyarrow
    python scripts/wrds_pull.py

Universe: common shares (shrcd 10, 11) on NYSE, AMEX and NASDAQ (exchcd
1, 2, 3), the MAX_UNIVERSE largest stocks by June market capitalisation
each year, kept for that calendar year. The window starts one year before
the news sample so the covariance lookback has history.

Outputs in data/: crsp_daily_2014_2024.parquet (date, permno, ticker, ret,
prc, shrout, me) and crsp_monthly_2014_2024.parquet (permno, date, ret).
The pricing scripts read the daily file.
"""
import os

import wrds

START, END = "2014-01-01", "2024-12-31"
MAX_UNIVERSE = 1000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DAILY_OUT = os.path.join(OUT_DIR, "crsp_daily_2014_2024.parquet")
MONTHLY_OUT = os.path.join(OUT_DIR, "crsp_monthly_2014_2024.parquet")

db = wrds.Connection()

daily = db.raw_sql(f"""
    select d.permno, d.date, d.ret, d.prc, d.shrout,
           n.ticker, n.shrcd, n.exchcd
    from crsp.dsf d
    join crsp.dsenames n
      on d.permno = n.permno
     and d.date between n.namedt and n.nameendt
    where d.date between '{START}' and '{END}'
      and n.shrcd in (10, 11)
      and n.exchcd in (1, 2, 3)
""", date_cols=["date"])

daily["me"] = daily["prc"].abs() * daily["shrout"]

# annual universe: the largest MAX_UNIVERSE stocks by June market cap, kept
# for that calendar year
daily["year"] = daily["date"].dt.year
june = daily[daily["date"].dt.month == 6]
me_june = june.groupby(["year", "permno"])["me"].last().reset_index()
keep = (me_june.sort_values(["year", "me"], ascending=[True, False])
        .groupby("year").head(MAX_UNIVERSE)[["year", "permno"]])
sel = daily.merge(keep.rename(columns={"year": "rank_year"}),
                  left_on=["permno", "year"], right_on=["permno", "rank_year"],
                  how="inner")

sel[["date", "permno", "ticker", "ret", "prc", "shrout", "me"]].to_parquet(
    DAILY_OUT, index=False)
print(f"saved: {DAILY_OUT}")

monthly = db.raw_sql(f"""
    select m.permno, m.date, m.ret
    from crsp.msf m
    join crsp.dsenames n
      on m.permno = n.permno
     and m.date between n.namedt and n.nameendt
    where m.date between '{START}' and '{END}'
      and n.shrcd in (10, 11)
      and n.exchcd in (1, 2, 3)
""", date_cols=["date"])
monthly.to_parquet(MONTHLY_OUT, index=False)
print(f"saved: {MONTHLY_OUT}")
print(f"daily rows: {len(sel):,}, monthly rows: {len(monthly):,}")
