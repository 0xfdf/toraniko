# toraniko

Toraniko is a complete implementation of a characteristic factor model in the same vein as Barra and Axioma.

The only dependencies are numpy and Polars. It supports market, sector and style factors; three styles are included: value, size and momentum. The functions you broadly want to have for constructing more style factors (or custom fundamental factors of any kind) are included.

This was developed as part of an education series of articles on quant trading: https://x.com/0xfdf/status/1808351541943763163. The model is substantially ready for production usage.