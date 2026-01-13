"""
Visualize backtest results
Display curves of total capital (cash + holdings), cumulative cost, and total assets over time
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Set font for displaying text
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_backtest_data(csv_path: str = "result/tsla_agent_real_profit_backtest_2022-01-03_gpt-4o-mini.csv"):
    """
    Load backtest results from CSV file
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with date as index
    """
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def load_llm_outputs_data(csv_path: str = None):
    """
    Load LLM outputs data from CSV file to get price and cash information
    
    Args:
        csv_path: Path to LLM outputs CSV file (if None, will try to infer from main CSV path)
    
    Returns:
        DataFrame with date as index, or None if file not found
    """
    if csv_path is None:
        return None
    
    # Try to infer LLM outputs path from main CSV path
    main_path = Path(csv_path)
    # Convert main CSV filename to LLM outputs filename
    # e.g., tsla_2018-01-03_gpt-4o-mini_1000000.csv -> tsla_llm_outputs_2018-01-03_gpt-4o-mini_1000000.csv
    if main_path.stem.startswith("tsla_"):
        parts = main_path.stem.split("_", 1)
        if len(parts) > 1:
            llm_outputs_filename = f"tsla_llm_outputs_{parts[1]}.csv"
            llm_outputs_path = main_path.parent / llm_outputs_filename
            if llm_outputs_path.exists():
                csv_path = str(llm_outputs_path)
            else:
                return None
        else:
            return None
    else:
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    except Exception as e:
        print(f"Warning: Could not load LLM outputs data: {e}")
        return None

def create_final_composition_charts(df: pd.DataFrame, csv_path: str = None, output_path: str = None):
    """
    Create Stacked Bar Chart and Pie Chart showing final composition (cost, holdings value, cash)
    
    Args:
        df: DataFrame containing backtest data
        csv_path: Path to the CSV file (used to extract metadata for filename)
        output_path: Path to save the output image (if None, will be auto-generated)
    """
    # Load LLM outputs data to get price and cash information
    llm_df = load_llm_outputs_data(csv_path)
    
    # Get final values
    final_market_value = df['market_value'].iloc[-1]
    final_cumulative_cost = df['cumulative_cost'].iloc[-1]
    final_hold_shares = df['hold_shares'].iloc[-1]
    
    # Calculate final holdings value and cash
    if llm_df is not None and len(llm_df) > 0:
        # Get the last date's price and cash from LLM outputs
        last_date = df.index[-1]
        # Find the closest date in LLM outputs
        if last_date in llm_df.index:
            final_price = llm_df.loc[last_date, 'current_price']
            final_cash = llm_df.loc[last_date, 'available_cash']
        else:
            # Use the last row if exact date not found
            final_price = llm_df['current_price'].iloc[-1]
            final_cash = llm_df['available_cash'].iloc[-1]
        final_holdings_value = final_hold_shares * final_price
    else:
        # Fallback: estimate from market_value
        # Assume holdings value is proportional to hold_shares
        # This is an approximation
        if final_hold_shares > 0:
            # Estimate price from market_value and shares
            # market_value = cash + shares * price
            # We'll use a simple approximation: assume cash is 10% of market_value
            estimated_cash = final_market_value * 0.1
            estimated_holdings_value = final_market_value - estimated_cash
            final_price = estimated_holdings_value / final_hold_shares if final_hold_shares > 0 else 0
            final_cash = estimated_cash
        else:
            final_holdings_value = 0
            final_price = 0
            final_cash = final_market_value
    
    # Ensure values are consistent: market_value = cash + holdings_value
    # Adjust if needed
    calculated_market_value = final_cash + final_holdings_value
    if abs(calculated_market_value - final_market_value) > 0.01:
        # Adjust cash to match market_value
        final_cash = final_market_value - final_holdings_value
    
    # If cash is negative, set it to 0 (as per user requirement)
    if final_cash < 0:
        final_cash = 0
    
    # Store values for display (cash is already >= 0)
    original_cash = final_cash
    original_holdings_value = max(0, final_holdings_value)
    original_cost = max(0, final_cumulative_cost)
    
    # For visualization, ensure all values are non-negative
    final_cash = original_cash  # Already >= 0
    final_holdings_value = original_holdings_value  # Already >= 0
    final_cumulative_cost = original_cost  # Already >= 0
    
    # Generate output path for composition charts
    if output_path:
        output_path_obj = Path(output_path)
        composition_output_path = output_path_obj.parent / (output_path_obj.stem + "_composition.png")
    else:
        fig_dir = Path("result") / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        if csv_path:
            csv_filename = Path(csv_path).stem
            if csv_filename.startswith("tsla_llm_outputs_"):
                base_name = csv_filename.replace("tsla_llm_outputs_", "tsla_")
            elif csv_filename.startswith("tsla_"):
                base_name = csv_filename
            else:
                base_name = csv_filename
            composition_output_path = fig_dir / (base_name + "_composition.png")
        else:
            composition_output_path = fig_dir / "backtest_composition.png"
    
    # Create figure with two subplots: Stacked Bar Chart and Pie Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stacked Bar Chart
    categories = ['Final Composition']
    cost_values = [original_cost]
    holdings_values = [original_holdings_value]
    cash_values = [original_cash]  # Already >= 0
    
    # Create stacked bar
    ax1.bar(categories, cost_values, label='Cumulative Cost', color='#A23B72', alpha=0.8)
    ax1.bar(categories, holdings_values, bottom=cost_values, label='Holdings Value', color='#2E86AB', alpha=0.8)
    ax1.bar(categories, cash_values, bottom=[cost_values[0] + holdings_values[0]], label='Cash', color='#06A77D', alpha=0.8)
    
    ax1.set_ylabel('Amount (USD)', fontsize=12)
    ax1.set_title('Final Composition: Cost, Holdings Value, and Cash', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    total_height = cost_values[0] + holdings_values[0] + cash_values[0]
    ax1.text(0, cost_values[0] / 2, f'${cost_values[0]:,.0f}', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax1.text(0, cost_values[0] + holdings_values[0] / 2, f'${holdings_values[0]:,.0f}', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax1.text(0, cost_values[0] + holdings_values[0] + cash_values[0] / 2, f'${cash_values[0]:,.0f}', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax1.text(0, total_height + total_height * 0.02, f'Total: ${total_height:,.0f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie Chart
    # Only include non-negative values for pie chart
    sizes = []
    labels = []
    colors_list = []
    explode_list = []
    
    total_positive = final_cumulative_cost + final_holdings_value + final_cash
    
    if total_positive > 0:
        if final_cumulative_cost > 0:
            cost_pct = (final_cumulative_cost / total_positive) * 100
            sizes.append(final_cumulative_cost)
            labels.append(f'Cumulative Cost\n${original_cost:,.0f}\n({cost_pct:.1f}%)')
            colors_list.append('#A23B72')
            explode_list.append(0.05)
        
        if final_holdings_value > 0:
            holdings_pct = (final_holdings_value / total_positive) * 100
            sizes.append(final_holdings_value)
            labels.append(f'Holdings Value\n${original_holdings_value:,.0f}\n({holdings_pct:.1f}%)')
            colors_list.append('#2E86AB')
            explode_list.append(0.05)
        
        if final_cash > 0:
            cash_pct = (final_cash / total_positive) * 100
            sizes.append(final_cash)
            labels.append(f'Cash\n${original_cash:,.0f}\n({cash_pct:.1f}%)')
            colors_list.append('#06A77D')
            explode_list.append(0.05)
        
        if len(sizes) > 0:
            explode_tuple = tuple(explode_list)
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_list, explode=explode_tuple,
                                               autopct='', startangle=90, textprops={'fontsize': 10})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Final Composition Distribution', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No positive values to display', ha='center', va='center', fontsize=12)
            ax2.set_title('Final Composition Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No positive values to display', ha='center', va='center', fontsize=12)
        ax2.set_title('Final Composition Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    composition_output_path_str = str(composition_output_path) if isinstance(composition_output_path, Path) else composition_output_path
    plt.savefig(composition_output_path_str, dpi=300, bbox_inches='tight')
    print(f"Composition charts saved to: {composition_output_path_str}")
    plt.show()

def create_time_series_stacked_chart(df: pd.DataFrame, csv_path: str = None, output_path: str = None):
    """
    Create a time series stacked area chart showing cost, holdings value, and cash over time
    
    Args:
        df: DataFrame containing backtest data
        csv_path: Path to the CSV file (used to extract metadata for filename)
        output_path: Path to save the output image (if None, will be auto-generated)
    """
    # Load LLM outputs data to get price and cash information over time
    llm_df = load_llm_outputs_data(csv_path)
    
    # Calculate holdings value and cash for each date
    if llm_df is not None and len(llm_df) > 0:
        # Merge dataframes on date index, using 'left' join to keep only df's index
        merged_df = df.join(llm_df[['current_price', 'available_cash']], how='left')
        # Forward fill missing values
        merged_df['current_price'] = merged_df['current_price'].ffill()
        merged_df['available_cash'] = merged_df['available_cash'].ffill()
        
        # Calculate holdings value for each date, using df.index explicitly
        # Use df.index to ensure we only get values for dates in df
        holdings_value = pd.Series(index=df.index, dtype=float)
        cash = pd.Series(index=df.index, dtype=float)
        
        for date in df.index:
            hold_shares = merged_df.loc[date, 'hold_shares']
            current_price = merged_df.loc[date, 'current_price']
            available_cash = merged_df.loc[date, 'available_cash']
            
            holdings_value.loc[date] = hold_shares * current_price if pd.notna(current_price) else 0
            # Set negative cash to 0
            cash.loc[date] = max(0, available_cash) if pd.notna(available_cash) else 0
    else:
        # Fallback: estimate from market_value
        # Assume cash is a percentage of market_value, adjust based on hold_shares
        # This is an approximation
        holdings_value = pd.Series(index=df.index, dtype=float)
        cash = pd.Series(index=df.index, dtype=float)
        
        for date in df.index:
            hold_shares = df.loc[date, 'hold_shares']
            market_value = df.loc[date, 'market_value']
            
            if hold_shares > 0:
                # Estimate: assume holdings are 90% of market_value when there are shares
                estimated_holdings_value = market_value * 0.9
                estimated_cash = market_value - estimated_holdings_value
            else:
                estimated_holdings_value = 0
                estimated_cash = market_value
            
            holdings_value.loc[date] = estimated_holdings_value
            # Set negative cash to 0
            cash.loc[date] = max(0, estimated_cash)
    
    # Get cumulative cost (already has df.index)
    cumulative_cost = df['cumulative_cost']
    
    # Ensure all Series have the same index and length as df.index
    # All should already be aligned, but ensure explicitly
    holdings_value = holdings_value.reindex(df.index, fill_value=0)
    cash = cash.reindex(df.index, fill_value=0)
    cumulative_cost = cumulative_cost.reindex(df.index, fill_value=0)
    
    # Generate output path for stacked chart
    if output_path:
        output_path_obj = Path(output_path)
        stacked_output_path = output_path_obj.parent / (output_path_obj.stem + "_stacked.png")
    else:
        fig_dir = Path("result") / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        if csv_path:
            csv_filename = Path(csv_path).stem
            if csv_filename.startswith("tsla_llm_outputs_"):
                base_name = csv_filename.replace("tsla_llm_outputs_", "tsla_")
            elif csv_filename.startswith("tsla_"):
                base_name = csv_filename
            else:
                base_name = csv_filename
            stacked_output_path = fig_dir / (base_name + "_stacked.png")
        else:
            stacked_output_path = fig_dir / "backtest_stacked.png"
    
    # Create figure for stacked area chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create stacked area chart
    ax.fill_between(df.index, 0, cumulative_cost, 
                    label='Cumulative Cost', color='#A23B72', alpha=0.7)
    ax.fill_between(df.index, cumulative_cost, cumulative_cost + holdings_value, 
                    label='Holdings Value', color='#2E86AB', alpha=0.7)
    ax.fill_between(df.index, cumulative_cost + holdings_value, 
                    cumulative_cost + holdings_value + cash, 
                    label='Cash', color='#06A77D', alpha=0.7)
    
    # Add line for total (market_value + cumulative_cost for reference)
    total_line = df['market_value'] + cumulative_cost
    ax.plot(df.index, total_line, 
            label='Total (Market Value + Cost)', 
            color='#000000', linewidth=2, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Amount (USD)', fontsize=12)
    ax.set_title('Composition Over Time: Cost, Holdings Value, and Cash', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    stacked_output_path_str = str(stacked_output_path) if isinstance(stacked_output_path, Path) else stacked_output_path
    plt.savefig(stacked_output_path_str, dpi=300, bbox_inches='tight')
    print(f"Time series stacked chart saved to: {stacked_output_path_str}")
    plt.show()

def visualize_backtest_results(df: pd.DataFrame, csv_path: str = None, output_path: str = None):
    """
    Visualize backtest results
    
    Args:
        df: DataFrame containing backtest data
        csv_path: Path to the CSV file (used to extract metadata for filename)
        output_path: Path to save the output image (if None, will be auto-generated)
    """
    # Generate output path if not provided
    if output_path is None:
        # Create result/fig folder if it doesn't exist
        fig_dir = Path("result") / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        if csv_path:
            # Extract filename from csv_path
            csv_filename = Path(csv_path).stem  # Get filename without extension
            # CSV filename format: tsla_{开始时间}_{模型名称}_{起始资金} or tsla_llm_outputs_{开始时间}_{模型名称}_{起始资金}
            # For visualization, we want: tsla_{开始时间}_{模型名称}_{起始资金}.png
            if csv_filename.startswith("tsla_llm_outputs_"):
                # Remove 'llm_outputs_' part to get the base name
                base_name = csv_filename.replace("tsla_llm_outputs_", "tsla_")
            elif csv_filename.startswith("tsla_"):
                # Already in correct format
                base_name = csv_filename
            else:
                # If format doesn't match, use the CSV filename as is
                base_name = csv_filename
            image_filename = base_name + ".png"
            output_path = fig_dir / image_filename
        else:
            # Fallback: use default name
            fig_dir = Path("result") / "fig"
            fig_dir.mkdir(parents=True, exist_ok=True)
            output_path = fig_dir / "backtest_visualization.png"
    else:
        # If output_path is provided, ensure the directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    # First subplot: Total capital and cumulative cost
    ax1 = axes[0]
    
    # Plot total capital (cash + holdings)
    ax1.plot(df.index, df['market_value'], 
             label='Total Capital (Cash + Holdings)', 
             linewidth=2, 
             color='#2E86AB')
    
    # Plot cumulative cost
    ax1.plot(df.index, df['cumulative_cost'], 
             label='Cumulative Cost', 
             linewidth=2, 
             color='#A23B72',
             linestyle='--')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Amount (USD)', fontsize=12)
    ax1.set_title('Total Capital and Cumulative Cost Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Second subplot: Total assets (total capital - cumulative cost) and cumulative net profit
    ax2 = axes[1]
    
    # Calculate total assets (total capital - cumulative cost)
    total_assets = df['market_value'] - df['cumulative_cost']
    
    # Plot total assets
    ax2.plot(df.index, total_assets, 
             label='Total Assets (Capital - Cost)', 
             linewidth=2, 
             color='#06A77D')
    
    # Plot cumulative net profit (for comparison)
    ax2.plot(df.index, df['cumulative_net_profit'], 
             label='Cumulative Net Profit', 
             linewidth=2, 
             color='#F18F01',
             linestyle=':')
    
    # Add zero line
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Amount (USD)', fontsize=12)
    ax2.set_title('Total Assets and Cumulative Net Profit Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Third subplot: Cumulative cost, cumulative net profit, and gross profit
    ax3 = axes[2]
    
    # Calculate initial cash (from first day's market value, as it should equal initial cash)
    initial_cash = df['market_value'].iloc[0]
    
    # Calculate gross profit (收益 = 现金+持仓-原始资金)
    gross_profit = df['market_value'] - initial_cash
    
    # Plot cumulative cost
    ax3.plot(df.index, df['cumulative_cost'], 
             label='Cumulative Cost', 
             linewidth=2, 
             color='#A23B72',
             linestyle='--')
    
    # Plot cumulative net profit
    ax3.plot(df.index, df['cumulative_net_profit'], 
             label='Cumulative Net Profit', 
             linewidth=2, 
             color='#F18F01',
             linestyle='-')
    
    # Plot gross profit (收益)
    ax3.plot(df.index, gross_profit, 
             label='Gross Profit (Cash + Holdings - Initial Capital)', 
             linewidth=2, 
             color='#2E86AB',
             linestyle='-.')
    
    # Add zero line
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Amount (USD)', fontsize=12)
    ax3.set_title('Cumulative Cost, Cumulative Net Profit, and Gross Profit Over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Format y-axis as currency
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    # Convert Path object to string for plt.savefig
    output_path_str = str(output_path) if isinstance(output_path, Path) else output_path
    plt.savefig(output_path_str, dpi=300, bbox_inches='tight')
    print(f"Visualization chart saved to: {output_path_str}")
    
    # Display statistics
    print("\n=== Backtest Statistics ===")
    print(f"Backtest start date: {df.index[0].strftime('%Y-%m-%d')}")
    print(f"Backtest end date: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Backtest days: {len(df)} days")
    print(f"\nInitial total capital: ${df['market_value'].iloc[0]:,.2f}")
    print(f"Final total capital: ${df['market_value'].iloc[-1]:,.2f}")
    print(f"Total capital change: ${df['market_value'].iloc[-1] - df['market_value'].iloc[0]:,.2f}")
    print(f"Total capital change rate: {(df['market_value'].iloc[-1] / df['market_value'].iloc[0] - 1) * 100:.2f}%")
    print(f"\nFinal cumulative cost: ${df['cumulative_cost'].iloc[-1]:,.2f}")
    print(f"Final cumulative net profit: ${df['cumulative_net_profit'].iloc[-1]:,.2f}")
    print(f"\nFinal total assets: ${total_assets.iloc[-1]:,.2f}")
    print(f"Total assets change: ${total_assets.iloc[-1] - total_assets.iloc[0]:,.2f}")
    print(f"Total assets change rate: {(total_assets.iloc[-1] / total_assets.iloc[0] - 1) * 100:.2f}%")
    
    plt.show()
    
    # Create additional visualizations: Stacked Bar Chart and Pie Chart
    create_final_composition_charts(df, csv_path, output_path)
    
    # Create time series stacked area chart
    create_time_series_stacked_chart(df, csv_path, output_path)

def main():
    """Main function"""
    csv_path = "result/tsla_2018-01-03_gpt-4o-mini_1000000.csv"
    
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File not found {csv_path}")
        print("Please run demo.py first to generate backtest results")
        return
    
    # Load data
    print(f"Loading data: {csv_path}...")
    df = load_backtest_data(csv_path)
    print(f"Data loaded successfully, {len(df)} records")
    
    # Visualize
    visualize_backtest_results(df, csv_path=csv_path)

if __name__ == "__main__":
    main()
