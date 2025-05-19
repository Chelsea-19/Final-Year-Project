import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Data Loading and Preprocessing
df = pd.read_excel("volume.xlsx")  

if df['ratio'].dtype == 'object':
    df['ratio'] = df['ratio'].str.rstrip('%').astype(float) / 100

# Identify left and right kidneys
df['side'] = df['ID'].apply(lambda x: 'Left' if '_L' in x else 'Right')
left = df[df['side'] == 'Left']
right = df[df['side'] == 'Right']

# 2. Basic Statistical Analysis
print("=" * 40)
print("=== Volume Error Statistics ===")
print(df['ratio'].describe(percentiles=[.25, .5, .75, .95]))

print("\n=== Global Geometric Accuracy Statistics ===")
print(df[['Chamfer', 'Hausdorff']].describe())

# Chamfer and Hausdorff means and standard deviations
print("\n=== 4.3.2 Global Geometric Similarity Metrics ===")
print(f"Chamfer Mean: {df['Chamfer'].mean():.4f} ± {df['Chamfer'].std():.4f}")
print(f"Hausdorff Mean: {df['Hausdorff'].mean():.4f} ± {df['Hausdorff'].std():.4f}")

# Pearson correlation between geometric accuracy and volume error
corr_chamfer, p_chamfer = stats.pearsonr(df['Chamfer'], df['ratio'])
corr_hausdorff, p_hausdorff = stats.pearsonr(df['Hausdorff'], df['ratio'])
print(f"\nCorrelation (Chamfer vs. Volume Error): r={corr_chamfer:.2f}, p={p_chamfer:.2f}")
print(f"Correlation (Hausdorff vs. Volume Error): r={corr_hausdorff:.2f}, p={p_hausdorff:.2f}")

# 3. Visualization Settings 
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'

# 4. Volume Error Distribution
fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
sns.histplot(data=df, x='ratio', bins=30, kde=True, ax=ax1)
ax1.set_title(f'Volume Error Distribution (Mean={df["ratio"].mean() * 100:.2f}%)')
ax1.axvline(df['ratio'].median(), color='red', linestyle='--', label='Median')
ax1.set_xlabel('Volume Error Ratio')
ax1.legend()

# 5. Geometric Accuracy vs. Error
ax2 = plt.subplot2grid((2, 3), (0, 2))
sns.scatterplot(data=df, x='Chamfer', y='ratio', hue='Hausdorff', 
                size='Hausdorff', sizes=(20, 200), ax=ax2)
ax2.set_title('Chamfer Distance vs. Volume Error')

plt.tight_layout()
plt.savefig('overview_error_analysis.png', dpi=300)

# 6. Geometric Distance Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Chamfer'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Chamfer Distance Distribution')
sns.histplot(df['Hausdorff'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Hausdorff Distance Distribution')
plt.tight_layout()
plt.savefig('geo_distribution.png', dpi=300)

# 7. Correlation Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.regplot(x='Chamfer', y='ratio', data=df, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Chamfer vs. Volume Error')
sns.regplot(x='Hausdorff', y='ratio', data=df, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Hausdorff vs. Volume Error')
plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300)

# 8. Parameter Sensitivity Analysis
print("\n=== 4.3.3 Parameter Sensitivity Analysis ===")
# Group by depth
depth_bins = [7, 8, 10, 12]
labels = ['7 layers', '8-10 layers', '12 layers']
df['depth_group'] = pd.cut(df['depth'], bins=depth_bins, labels=labels, right=False)
depth_error = df.groupby('depth_group')['ratio'].mean()
print("\nMean Volume Error by Depth Group:")
print(depth_error)

# Correlation between scale and volume error
corr_scale, p_scale = stats.pearsonr(df['scale'], df['ratio'])
print(f"\nCorrelation (Scale vs. Volume Error): r={corr_scale:.2f}, p={p_scale:.2f}")

# Barplot: Volume error by depth group
plt.figure(figsize=(8, 5))
sns.barplot(x='depth_group', y='ratio', data=df, palette="Blues_d")
plt.title('Volume Error by Network Depth')
plt.ylabel('Volume Error (%)')
plt.savefig('depth_analysis.png', dpi=300)

# Scatterplot: Scale vs. Volume Error
plt.figure(figsize=(8, 5))
sns.scatterplot(x='scale', y='ratio', data=df, hue='depth', palette="viridis")
plt.title('Scale vs. Volume Error')
plt.axvspan(1.4, 1.5, color='green', alpha=0.1, label='Optimal Range')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout() 
plt.savefig('scale_analysis.png', dpi=300, bbox_inches='tight')  

# 9. Left vs Right Kidney Comparison
print("\n=== 4.3.4 Error Distribution between Left and Right Kidneys ===")
metrics = ['ratio', 'Chamfer', 'Hausdorff']
results = []
for metric in metrics:
    t_stat, p_val = stats.ttest_ind(left[metric], right[metric])
    results.append({
        'Metric': metric,
        'Left Mean': left[metric].mean(),
        'Right Mean': right[metric].mean(),
        'p-value': p_val
    })
results_df = pd.DataFrame(results)
print("\nComparison of Metrics between Left and Right Kidneys:")
print(results_df)

# Boxplot comparison
plt.figure(figsize=(10, 6))
melt_df = df.melt(id_vars=['side'], value_vars=metrics)
sns.boxplot(x='variable', y='value', hue='side', data=melt_df, palette="Set2")
plt.title('Comparison of Errors between Left and Right Kidneys')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.savefig('side_comparison.png', dpi=300)

plt.close('all')
print("\nAnalysis completed. All charts saved in the current directory.")
