# 📈 Chat2Plot - interactive text-to-visualization with LLM

Chat2plot is a project that provides visualizations based on chat instructions for given data.

demo: https://chat2plot-sample.streamlit.app/

## Usage

```Python
import os
import pandas as pd
from chat2plot import chat2plot

# 1. Set api-key
os.environ["OPENAI_API_KEY"] = "..."

df = pd.read_csv(...)

# 2. Pass a dataframe to draw
c2p = chat2plot(df)

# 3. Plot chart interactively
c2p("average target over countries")

c2p("change to horizontal-bar chart")

c2p("...")
```

## Why Chat2Plpot

Inside Chat2Plot, LLM does not generate Python code,
but generates plot specifications in json.

The declarative visualization specification in json is transformed into actual charts in 
Chat2Plot using plotly or altair, but users can also use json directly in their own applications.

This design limits the visualization expression compared to Python code generation 
(such as ChatGPT's Code Interpreter Plugin), but has the following practical advantages:

- Secure
    - More secure execution, as LLM does not directly generate code.

- Language-independent
    - Declarative data structures are language-agnostic, making it easy to plot in non-Python environments.

- Interactive
    - Declarative data can be modified by the user to improve plots through collaborative work between the user and LLM.

The json schema can be selected from a default simple definition or a vega-lite compliant schema.

```Python
c2p = chat2plot(df, "vega")  # use vega-lite format

ret = c2p("plot x vs y")

ret.config  # get vega-lite compliant json data
```
