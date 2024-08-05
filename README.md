# toraniko

Toraniko is a complete implementation of a risk model suitable for quantitative and systematic trading at institutional scale. In particular, it is a characteristic factor model in the same vein as Barra and Axioma (in fact, given the same datasets, it approximately reproduces Barra's estimated factor returns).

![mom_factor](https://github.com/user-attachments/assets/f9d2927c-e899-4fd6-944c-8f9a104b410f)

Using this library, you can create new custom factors and estimate their returns. Then you can estimate a factor covariance matrix suitable for portfolio optimization with factor exposure constraints (e.g. to main a market neutral portfolio).

The only dependencies are numpy and polars. It supports market, sector and style factors; three styles are included: value, size and momentum. The library also comes with generalizable math and data cleaning utility functions you'd want to have for constructing more style factors (or custom fundamental factors of any kind).

## Installation

Using pip:

`pip install toraniko`

## User Manual

#### Data

You'll need the following data to run a complete model estimation:

1. Sector scores, used for estimating the market and sector factor returns. GICS level 1 is suitable for this. The sector scores consist of one row per asset, with 0s in each column except for the sector in which the asset is a member, which is filled with 1.

```
symbol	Basic Materials	Communication Services	Consumer Cyclical	Consumer Defensive	Energy	Financial Services	Healthcare	Industrials	Real Estate	Technology	Utilities
str	i64	i64	i64	i64	i64	i64	i64	i64	i64	i64	i64
"A"	0	0	0	0	0	0	1	0	0	0	0
"AA"	1	0	0	0	0	0	0	0	0	0	0
"AACI"	0	0	0	0	0	1	0	0	0	0	0
"AACT"	0	0	0	0	0	1	0	0	0	0	0
"AADI"	0	0	0	0	0	0	1	0	0	0	0
…	…	…	…	…	…	…	…	…	…	…	…
"ZVRA"	0	0	0	0	0	0	1	0	0	0	0
"ZVSA"	0	0	0	0	0	0	1	0	0	0	0
"ZWS"	0	0	0	0	0	0	0	1	0	0	0
"ZYME"	0	0	0	0	0	0	1	0	0	0	0
"ZYXI"	0	0	0	0	0	0	1	0	0	0	0
```

2. Symbol-by-symbol daily asset returns for a large universe of equities:

```
date	symbol	asset_returns
date	str	f64
2013-01-02	"A"	0.022962
2013-01-02	"AAMC"	-0.073171
2013-01-02	"AAME"	0.035566
2013-01-02	"AAON"	0.019163
2013-01-02	"AAP"	0.001935
…	…	…
2024-02-23	"ZVRA"	-0.025
2024-02-23	"ZVSA"	0.291311
2024-02-23	"ZWS"	0.006378
2024-02-23	"ZYME"	0.000838
2024-02-23	"ZYXI"	0.001552
```

3. For the value factor: symbol-by-symbol daily market cap, cash flow, share count, revenue and book value estimates, so you can calculate book-price, sales-price and cash flow-price metrics:

```
date	symbol	book_price	sales_price	cf_price	market_cap
date	str	f64	f64	f64	f64
2013-10-30	"AAPL"	0.343017	0.081994	0.007687	4.5701e11
2013-10-31	"AAPL"	0.342231	0.081763	0.007665	4.5830e11
2013-11-01	"AAPL"	0.341398	0.081521	0.007643	4.5966e11
2013-11-04	"AAPL"	0.340947	0.08137	0.007628	4.6051e11
2013-11-05	"AAPL"	0.340491	0.081219	0.007614	4.6137e11
…	…	…	…	…	…
2024-02-16	"AAPL"	0.072243	0.040937	0.001972	2.9209e12
2024-02-20	"AAPL"	0.072405	0.040942	0.001972	2.9206e12
2024-02-21	"AAPL"	0.072614	0.040973	0.001974	2.9184e12
2024-02-22	"AAPL"	0.072792	0.040987	0.001974	2.9174e12
2024-02-23	"AAPL"	0.073007	0.041021	0.001976	2.9150e12
```

#### Style factor score calculation

Taking the foregoing data together you'll have:

```
date	symbol	book_price	sales_price	cf_price	market_cap	asset_returns
date	str	f64	f64	f64	f64	f64
2013-02-12	"A"	null	null	null	1.1080e10	0.00045
2013-02-12	"AAON"	null	null	null	5.5410e8	0.006741
2013-02-12	"AAP"	null	null	null	5.4212e9	0.002679
2013-02-12	"AAPL"	0.302543	0.1189	null	4.5847e11	-0.025069
2013-02-12	"AAT"	null	null	null	1.1135e9	0.006764
…	…	…	…	…	…	…
2024-02-23	"ZS"	0.105832	0.016538	0.007052	3.5311e10	0.04015
2024-02-23	"ZTS"	0.125734	0.025145	-0.017418	8.8010e10	0.002797
2024-02-23	"ZUO"	0.424385	0.089243	0.073274	1.2309e9	0.002448
2024-02-23	"ZWS"	0.296327	0.067669	0.001916	5.2727e9	0.006378
2024-02-23	"ZYME"	0.363093	0.016538	-0.054523	7.3077e8	0.000838
```

Then to estimate the momentum factor, you can run

```
from toraniko.styles import factor_mom

mom_df = factor_mom(df.select("symbol", "date", "asset_returns"), trailing_days=252, winsor_factor=0.01).collect()
```

and you'll obtain scores roughly resembling this histogram:

<img width="598" alt="Screenshot 2024-08-05 at 12 28 39 AM" src="https://github.com/user-attachments/assets/88983248-a982-4c9e-9048-c01f1e7d191a">

Likewise for value, you can run: 

```
from toraniko.styles import factor_val

value_df = factor_val(df.select("date", "symbol", "book_price", "sales_price", "cf_price")).collect()
```

Similarly:

<img width="624" alt="Screenshot 2024-08-05 at 12 30 08 AM" src="https://github.com/user-attachments/assets/ca5c1afc-128e-4cd6-9871-6d7eb0e77ebc">

#### Factor return estimation

Let's say you've estimated three style factors: value, momentum, size.

```
date	symbol	mom_score	sze_score	val_score
date	str	f64	f64	f64
2014-03-07	"A"	0.793009	-0.872847	0.265801
2014-03-07	"AAON"	1.128932	0.939116	-1.050307
2014-03-07	"AAP"	2.190209	0.0	0.351273
2014-03-07	"AAPL"	-0.202091	-3.373659	-0.289713
2014-03-07	"AAT"	-0.413211	0.805413	0.052482
…	…	…	…	…
2024-02-23	"ZS"	2.783905	-1.303952	-1.778753
2024-02-23	"ZTS"	0.030749	-1.905171	-1.619675
2024-02-23	"ZUO"	0.235533	0.905671	0.253427
2024-02-23	"ZWS"	0.594781	0.0	-0.520947
2024-02-23	"ZYME"	-1.631452	1.248919	-1.451382
```

Merge these with the aforementioned GICS sector scores, and take the top N by market cap each day on a suitable universe. Here we'll do the Russell 3000:

```
from toraniko.utils import top_n_by_group

ddf = (
    ret_df.join(cap_df.drop("book_value"), on=["date", "symbol"])
    .join(sector_scores, on="symbol")
    .join(style_scores, on=["date", "symbol"])
    .drop_nulls()
)
ddf = (
    top_n_by_group(
        ddf.lazy(),
        3000,
        "market_cap",
        ("date",),
        True
    )
    .collect()
    .sort("date", "symbol")
)

returns_df = ddf.select("date", "symbol", "asset_returns")
mkt_cap_df = ddf.select("date", "symbol", "market_cap")
sector_df = ddf.select(["date"] + list(sector_scores.columns))
style_df = ddf.select(style_scores.columns)
```

Then simply:

```
from toraniko.model import estimate_factor_returns

fac_df, eps_df = estimate_factor_returns(returns_df, mkt_cap_df, sector_df, style_df, winsor_factor=0.1, residualize_styles=False)
```

On an M1 MacBook, this estimates 10+ years of daily market, sector and style factor returns in under a minute.

Here is a comparison of the model value factor out versus Barra's. Even on a relatively low quality data source (Yahoo Finance) and without significant effort in cleaning corporate actions, the results are comparable over a 10 year period:

![val_factor](https://github.com/user-attachments/assets/28f41989-f802-4c2f-beed-1d2bda24a96d)

![valu](https://github.com/user-attachments/assets/366f49a8-d7e7-46de-bb61-6f656393275a)
