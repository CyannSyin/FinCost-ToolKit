import matplotlib.pyplot as plt


def plot_cost_pie(
    commission_total,
    token_total,
    infra_total,
    monthly_total,
    uncertain_total,
    output_path,
    include_monthly=True,
):
    labels = ["commission", "token", "infra", "monthly", "uncertain"]
    values = [commission_total, token_total, infra_total, monthly_total, uncertain_total]
    colors = ["#809bce", "#eac4d5", "#b8e0d4", "#95b8d1", "#f5e2ea"]
    if not include_monthly:
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if l != "monthly"]
        labels, values, colors = zip(*filtered) if filtered else ([], [], [])

    total = sum(values)
    if total <= 0:
        print("No costs to plot.")
        return
    formatted_labels = [f"{label} (${value:.2f})" for label, value in zip(labels, values)]

    def make_autopct(vals):
        def _autopct(pct):
            value = pct * sum(vals) / 100.0
            return f"{pct:.1f}%\n${value:.2f}"

        return _autopct

    plt.figure(figsize=(6, 6))
    plt.pie(
        values,
        labels=formatted_labels,
        autopct=make_autopct(values),
        startangle=90,
        colors=colors,
    )
    plt.title("Cost share: commission vs token vs infra")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved pie chart to: {output_path}")


def plot_performance_lines(dates, holding_profit_series, cumulative_cost_series, real_profit_series, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(
        dates,
        holding_profit_series,
        label="Holding Profit (holdings + cash - initial cash)",
        color="#88a4c9",
    )
    plt.plot(
        dates,
        cumulative_cost_series,
        label="Cumulative Cost (token + infra + commission)",
        color="#ff8696",
    )
    plt.plot(
        dates,
        real_profit_series,
        label="Real Profit (holding profit - cost)",
        color="#ff8531",
    )
    plt.axhline(0, color="black", linewidth=2.5, linestyle="--")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved performance chart to: {output_path}")
