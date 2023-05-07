# 📈 Chat2Plot - interactive text-to-visualization with LLM

Chat2plot is a project that provides visualizations based on chat instructions for given data.

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

cp2("...")
```

## Why Chat2Plpot

Inside Chat2Plot, LLM does not generate Python code, but rather a declarative data structure representing the plot specification.

While this design has a lower capacity of plots that can be generated compared to Python code (e.g., ChatGPT's Code Interpreter Plugin), it has the following practical advantages:

- Secure
    - More secure execution, as LLM does not directly generate code.

- Language-independent
    - Declarative data structures are language-independent, making it easy to plot in non-Python environments.

- Interactive
    - Declarative data can be modified by the user to improve plots through collaborative work between the user and LLM.

This declarative data structure can be chosen between default (simple dataclass) and the vega-lite format.

```Python
c2p = chat2plot(df, "vega")  # use vega-lite format
```
