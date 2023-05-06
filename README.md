# 📈 Chat2Plot - interactive text-to-visualization with LLM

Chat2plot is a project that provides visualizations based on chat instructions for given data.

## Usage

```Python
import pandas as pd
from chat2plot import Chat2Plot

# 1. Set api-key beforehand
# os.environ["OPENAI_API_KEY"] = "..."

df = pd.read_csv(...)

# 2. Pass dataframe to draw
c2p = Chat2Plot(df)

# 3. Plot chart interactively
c2p("average target over countries")

c2p("change to horizontal-bar chart")

cp2("...")
```
