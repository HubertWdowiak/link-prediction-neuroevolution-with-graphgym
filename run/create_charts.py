import pandas as pd
import os
import json

pd.options.plotting.backend = "plotly"

rows = []
result_dir = os.path.join("wyniki", "example")
for neuro_epoch in sorted(os.listdir(result_dir)):
    if neuro_epoch == "config.yaml":
        continue
    for genome_dir in sorted(os.listdir(os.path.join(result_dir, neuro_epoch))):
        with open(
            os.path.join(result_dir, neuro_epoch, genome_dir, "agg", "val", "best.json")
        ) as file:
            row = json.load(file)
        row["neuro_epoch"] = neuro_epoch
        rows.append(row)
df = pd.DataFrame.from_records(rows)

fig = df[["neuro_epoch", "loss"]].groupby("neuro_epoch").min().plot.line()
fig.update_layout(
    xaxis_title="Neuro Epoch",
    yaxis_title="Min loss in population",
    yaxis=dict(
        title="Min loss in Population",
        tickformat=".2f",
    ),
)
fig.show()

fig = df[["neuro_epoch", "epoch"]].groupby("neuro_epoch").mean().plot.line()
fig.update_layout(
    xaxis_title="Neuro Epoch",
    yaxis_title="Mean number of epochs in population",
    yaxis=dict(
        title="Mean number of epochs in Population",
        tickformat=".2f",
    ),
)
fig.show()

fig = (
    df[["neuro_epoch", "accuracy", "f1", "auc", "loss"]]
    .groupby("neuro_epoch")
    .mean()
    .plot.line()
)
fig.update_layout(
    xaxis_title="Neuro Epoch",
    yaxis_title="Aggregated Value",
    yaxis=dict(
        title="Aggregated Value in Population",
        tickformat=".2f",
    ),
)
fig.update_traces(name="Mean Accuracy", selector=dict(name="accuracy"))
fig.update_traces(name="Mean F1", selector=dict(name="f1"))
fig.update_traces(name="Mean AUC", selector=dict(name="auc"))
fig.update_traces(name="Mean Loss", selector=dict(name="loss"))

fig.show()
