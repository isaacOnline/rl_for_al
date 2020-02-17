import plotly.express as px
import plotly
import pandas as pd

def plot_policy():
    data = pd.read_csv("results/policy_by_ts_tt_ratio.csv")
    Ts = 1
    data["Tt/Ts"] = data.Tt / Ts
    consolidated = data.iloc[1::100]
    fig = px.scatter(consolidated,
                     x="S",
                     y="fout",
                     animation_frame="Tt/Ts")
    fig.update_layout(
        xaxis_title = "Length of Hypothesis Space",
        yaxis_title = "Policy"
    )
    fig.show()
    plotly.io.write_html(fig, file='policies_by_time_ratio.html')