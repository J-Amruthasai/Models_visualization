import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_bar_chart(results_df, metric: str):
    """
    Plots a bar chart comparing models on a specific metric.
    """
    fig, ax = plt.subplots()
    sns.barplot(x=results_df.index, y=results_df[metric], palette="viridis", ax=ax)
    ax.set_title(f"{metric} Comparison")
    ax.set_ylabel(metric)
    ax.set_xlabel("Models")
    plt.xticks(rotation=45)
    return fig
