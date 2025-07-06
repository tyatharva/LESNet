# Values (calculate from spreadsheets)
values = np.array([
    [86.19, 124.95, 100.55, 53.97],
    [113.57, 112.38, 109.46, 89.69],
    [43.43, 32.46, 33.93, 78.98],
    
    [89.68, 124.58, 105.60, 51.51],
    [91.16, 90.98, 104.53, 88.67],
    [10.13, 18.67, 25.88, 71.35]
])

# Create lake labels inside cells
lake_labels = np.array([
    ["Superior\n86.19", "Michigan\n124.95", "Erie\n100.55", "Ontario\n53.97"],
    ["Superior\n113.57", "Michigan\n112.38", "Erie\n109.46", "Ontario\n89.69"],
    ["Superior\n43.43", "Michigan\n32.46", "Erie\n33.93", "Ontario\n78.98"],
    
    ["Superior\n89.68", "Michigan\n124.58", "Erie\n105.60", "Ontario\n51.51"],
    ["Superior\n91.16", "Michigan\n90.98", "Erie\n104.53", "Ontario\n88.67"],
    ["Superior\n10.13", "Michigan\n18.67", "Erie\n25.88", "Ontario\n71.35"]
])

# Reshape for 2 rows × 3 columns format
values = values.reshape(2, 3, 2, 2)
lake_labels = lake_labels.reshape(2, 3, 2, 2)

# Plotting
fig, ax = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

# Iterate through grid cells
for i in range(2):  # Neighborhoods (rows)
    for j in range(3):  # Thresholds (columns)
        sns.heatmap(values[i, j], annot=lake_labels[i, j], fmt='',
                     cmap='RdYlGn', center=0, cbar=False,
                     xticklabels=False, yticklabels=False, ax=ax[i, j]) #, square=True

        # Remove grid ticks and manually add text for labels
        ax[i, j].set_xticks([])  # Remove x-axis ticks
        ax[i, j].set_yticks([])  # Remove y-axis ticks

# Add numeric labels for the grid
thresholds = ['Light (0.25)', 'Moderate (1.00)', 'Heavy (2.00)']
neighborhoods = ['LESNet-A', 'LESNet-B']

for a in range(3):  # Threshold values
    ax[1, a].text(0.5, -0.05, thresholds[a], ha='center', va='center', fontsize=10, transform=ax[1, a].transAxes)

for a in range(2):  # Neighborhood values
    ax[a, 0].text(-0.05, 0.5, neighborhoods[a], ha='center', va='center', fontsize=12, transform=ax[a, 0].transAxes, rotation=90)

# Add axis labels
fig.supxlabel('Threshold (mm)', fontsize=12)

# Overall Title
fig.suptitle('Percent Improvement of Model Over HRRR in Fractions Skill Score', fontsize=14)

# plt.savefig("fss.pdf", dpi=600, bbox_inches="tight")
plt.show()
