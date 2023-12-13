# `tinyhnsw` + `sentence-transformers`

Use the index you just built to search for text from documents!


## Installation

From `pip`:

```sh
pip install tinyhnsw[examples]
```

Or locally:

```sh
poetry install --all-extras
```

## Usage

```sh
python text_retrieval.py
```

Which will search for the nearest sentences from the Financial Phrasebank dataset.
For instance, the search for `positive returns` yields:

```
1. Return on investment was 5.0 % , compared to a negative 4.1 % in 2009 .

2. As a result , the company currently anticipates net sales to increase and the operating result to be positive .

3. The company now estimates its net sales in 2010 to increase considerably from 2009 and its operating result to be clearly positive .

4. For the current year , Raute expects its net sales to increase and the operating result -- to be positive .

5. Nordea sees a return to positive growth for the Baltic countries in 2011 .
```