import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dummy.csv", index_col=0)
dff = df[(df.Avg_Translational_Velocity > 2) & (df.Total_Angular_Displacement > 10)]
dff = dff.dropna()
dff = dff[dff.Pre_Stim_Velocity > 4]
dff.loc[:, ["Perturbation"]] = dff["Perturbation"].str.split("_", expand=True)[0]
dictionary = {
"tnt": "TNT",
"cg16974": "CG16974",
"cg32055": "CG32055",
"side-viii": "SIDE-VIII",
"beat-vii": "Beat-VII",
"cg43795": "CG43795",
"fas3": "Fas3",
"cg9394": "CG9394",
"cg32206": "CG32206",
"nrx-1": "Nrx-1",
"beat-iv": "Beat-IV",
"cg5888": "CG5888",
"cg31862": "CG31862",
"empty": "wild-type"
}
dff.replace(dictionary, inplace=True)
orders_avg = dff.groupby(["Perturbation"])["Avg_Angular_Velocity"].median().sort_values()
orders_total = dff.groupby(["Perturbation"])["Total_Angular_Displacement"].median().sort_values()
f, ax = plt.subplots(figsize=(8,8))
sns.set(style="darkgrid", rc={"lines.linewidth": 2, "font.family": "Carlito", "font.size": 24, "axes.labelsize": 24})
sns.stripplot(y="Perturbation", x="Total_Angular_Displacement", data=dff, order=list(orders_avg.index), color="black", size=3, ax=ax)
sns.boxplot(y="Perturbation", x="Total_Angular_Displacement", width=.5, data=dff, order=list(orders_total.index), color="gray", showfliers=False, ax=ax)
ax.set_xlabel("Average angular velocity (degrees per second)", {"fontsize": "x-small"})
ax.set_ylabel("RNAi target gene", {"fontsize": "x-small"})
plt.tight_layout()
plt.show()
# sns.stripplot(y="Perturbation", x="Total_Angular_Displacement", data=dff, order=list(orders_avg.index))
# sns.boxplot(y="Perturbation", x="Total_Angular_Displacement", data=dff, order=list(orders_total.index))
# plt.show()
