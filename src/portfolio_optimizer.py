"""
Portfolio optimization using Modern Portfolio Theory for Task 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Class for portfolio optimization using MPT"""
    
    def __init__(self, returns_data, expected_returns=None):
        """
        Initialize with historical returns data
        
        Parameters:
        -----------
        returns_data: DataFrame with assets as columns and returns as rows
        expected_returns: Dictionary of expected returns for each asset
        """
        self.returns_data = returns_data
        self.assets = returns_data.columns.tolist()
        self.num_assets = len(self.assets)
        
        if expected_returns is None:
            # Use historical mean as expected returns
            self.expected_returns = returns_data.mean() * 252  # Annualize
        else:
            self.expected_returns = pd.Series(expected_returns)
            
        self.cov_matrix = returns_data.cov() * 252  # Annualize covariance
        
    def calculate_portfolio_stats(self, weights):
        """Calculate portfolio return and volatility for given weights"""
        weights = np.array(weights)
        port_return = np.sum(self.expected_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Assuming risk-free rate of 2%
        risk_free_rate = 0.02
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        return {
            'return': port_return,
            'volatility': port_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio(self, target_return=None, target_risk=None):
        """Optimize portfolio for given target return or risk"""
        
        def negative_sharpe(weights):
            stats = self.calculate_portfolio_stats(weights)
            return -stats['sharpe_ratio']
        
        def portfolio_variance(weights):
            return self.calculate_portfolio_stats(weights)['volatility']**2
        
        def portfolio_return(weights):
            return self.calculate_portfolio_stats(weights)['return']
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum to 1
        
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
            
        if target_risk is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: self.calculate_portfolio_stats(x)['volatility'] - target_risk})
        
        # Bounds (0-1 for each asset)
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        # Optimize for maximum Sharpe ratio
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            stats = self.calculate_portfolio_stats(optimal_weights)
            return optimal_weights, stats
        else:
            raise ValueError("Optimization failed")
    
    def generate_efficient_frontier(self, num_portfolios=10000):
        """Generate random portfolios for efficient frontier"""
        results = []
        weights_record = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(self.num_assets)
            weights = weights / np.sum(weights)
            
            # Calculate portfolio statistics
            stats = self.calculate_portfolio_stats(weights)
            
            results.append([stats['return'], stats['volatility'], stats['sharpe_ratio']])
            weights_record.append(weights)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe_Ratio'])
        weights_df = pd.DataFrame(weights_record, columns=self.assets)
        
        # Find optimal portfolios
        max_sharpe_idx = results_df['Sharpe_Ratio'].idxmax()
        min_vol_idx = results_df['Volatility'].idxmin()
        
        max_sharpe_port = {
            'weights': weights_df.iloc[max_sharpe_idx],
            'return': results_df.iloc[max_sharpe_idx]['Return'],
            'volatility': results_df.iloc[max_sharpe_idx]['Volatility'],
            'sharpe_ratio': results_df.iloc[max_sharpe_idx]['Sharpe_Ratio']
        }
        
        min_vol_port = {
            'weights': weights_df.iloc[min_vol_idx],
            'return': results_df.iloc[min_vol_idx]['Return'],
            'volatility': results_df.iloc[min_vol_idx]['Volatility'],
            'sharpe_ratio': results_df.iloc[min_vol_idx]['Sharpe_Ratio']
        }
        
        return results_df, max_sharpe_port, min_vol_port
    
    def plot_efficient_frontier(self, results_df, max_sharpe_port, min_vol_port, save_path=None):
        """Plot the efficient frontier"""
        plt.figure(figsize=(12, 8))
        
        # Plot random portfolios
        plt.scatter(results_df['Volatility'], results_df['Return'],
                   c=results_df['Sharpe_Ratio'], cmap='viridis',
                   alpha=0.6, s=10)
        plt.colorbar(label='Sharpe Ratio')
        
        # Plot max Sharpe portfolio
        plt.scatter(max_sharpe_port['volatility'], max_sharpe_port['return'],
                   color='red', s=200, marker='*', label='Max Sharpe Ratio')
        
        # Plot min volatility portfolio
        plt.scatter(min_vol_port['volatility'], min_vol_port['return'],
                   color='green', s=200, marker='*', label='Min Volatility')
        
        plt.xlabel('Volatility (Annualized)', fontsize=12)
        plt.ylabel('Return (Annualized)', fontsize=12)
        plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text annotations
        plt.annotate(f"Sharpe: {max_sharpe_port['sharpe_ratio']:.2f}\n"
                    f"Vol: {max_sharpe_port['volatility']:.2%}\n"
                    f"Return: {max_sharpe_port['return']:.2%}",
                    xy=(max_sharpe_port['volatility'], max_sharpe_port['return']),
                    xytext=(max_sharpe_port['volatility'] + 0.05, max_sharpe_port['return']),
                    fontsize=9)
        
        plt.annotate(f"Sharpe: {min_vol_port['sharpe_ratio']:.2f}\n"
                    f"Vol: {min_vol_port['volatility']:.2%}\n"
                    f"Return: {min_vol_port['return']:.2%}",
                    xy=(min_vol_port['volatility'], min_vol_port['return']),
                    xytext=(min_vol_port['volatility'] + 0.05, min_vol_port['return'] - 0.02),
                    fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_covariance_heatmap(self, save_path=None):
        """Plot covariance matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.cov_matrix, dtype=bool))
        
        sns.heatmap(self.cov_matrix,
                   annot=True,
                   fmt='.4f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8},
                   mask=mask)
        
        plt.title('Covariance Matrix (Annualized)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def recommend_portfolio(self, risk_tolerance='medium'):
        """Recommend portfolio based on risk tolerance"""
        # Get optimal portfolios
        results_df, max_sharpe_port, min_vol_port = self.generate_efficient_frontier(num_portfolios=5000)
        
        if risk_tolerance == 'low':
            recommended_port = min_vol_port
            recommendation = "Conservative investor: Minimum Volatility Portfolio"
        elif risk_tolerance == 'high':
            recommended_port = max_sharpe_port
            recommendation = "Aggressive investor: Maximum Sharpe Ratio Portfolio"
        else:  # medium
            # Find portfolio with balanced risk-return
            balanced_idx = ((results_df['Sharpe_Ratio'] - results_df['Sharpe_Ratio'].mean()).abs()).idxmin()
            balanced_port = {
                'weights': pd.Series(np.random.dirichlet(np.ones(self.num_assets), size=1)[0], index=self.assets),
                'return': results_df.iloc[balanced_idx]['Return'],
                'volatility': results_df.iloc[balanced_idx]['Volatility'],
                'sharpe_ratio': results_df.iloc[balanced_idx]['Sharpe_Ratio']
            }
            recommended_port = balanced_port
            recommendation = "Moderate investor: Balanced Portfolio"
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Asset': self.assets,
            'Weight': recommended_port['weights'].values,
            'Expected_Return': self.expected_returns.values
        })
        
        portfolio_summary = {
            'Total_Expected_Return': recommended_port['return'],
            'Total_Volatility': recommended_port['volatility'],
            'Sharpe_Ratio': recommended_port['sharpe_ratio'],
            'Weights': summary,
            'Recommendation': recommendation
        }
        
        return portfolio_summary