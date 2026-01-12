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
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
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

def main():
    """Main function"""
    csv_path = "result/tsla_2022-01-03_gpt-4o-mini_50000.csv"
    
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
