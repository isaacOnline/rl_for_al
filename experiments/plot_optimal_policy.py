import plotly.express as px
import plotly
import pandas as pd
import numpy as np
from agents import UniformAgent

def run():
    outfile = "results/policy_by_ts_tt_ratio.csv"
    for Tt in [0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5] + list(range(980,1001)):
        agt = UniformAgent(Ts=1,Tt=1,N=2500)
        agt.plot_policy()
        # Put in code here to save


def plot_policy():
    data = pd.read_csv("results/policy_by_ts_tt_ratio.csv")
    Ts = 1
    data["Tt/Ts"] = data.Tt / Ts
    allowable_Tt = list(range(100)) + list(range(100,1001,50))
    Tt_slicer = np.isin(data.Tt, allowable_Tt)
    consolidated = data[Tt_slicer]

    S_slicer = np.isin(consolidated.S, consolidated.S.unique()[::10])
    consolidated = consolidated[S_slicer]

    fig = px.scatter(consolidated,
                     x="S",
                     y="fout",
                     animation_frame="Tt/Ts")
    fig.update_layout(
        xaxis_title = "Length of Hypothesis Space",
        yaxis_title = "Movement",
        title = "Policy by Tt/Ts Ratio",
        yaxis_range = [0,1]
    )
    plotly.io.write_html(fig, file='visualizations/policies_by_time_ratio.html')