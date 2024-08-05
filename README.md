# toraniko

Toraniko is a complete implementation of a risk model for quantitative and systematic trading. In particular, it is a characteristic factor model in the same vein as Barra and Axioma. It is basic but safe for production usage; in fact it is already being used in production at a systematic trading firm (>$2B GMV).

Using this library, you can create new custom factors and estimate their returns. From there you will be able to estimate a factor covariance matrix suitable for portfolio optimization with factor risk constraints (e.g. to main a market neutral portfolio).

The only dependencies are numpy and polars. It supports market, sector and style factors; three styles are included: value, size and momentum. The functions you broadly want to have for constructing more style factors (or custom fundamental factors of any kind) are included.

