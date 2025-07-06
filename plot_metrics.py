# Values (calculate from spreadsheets)
model1 = np.array([
    [48.38, 61.60, 48.90, 61.02],
    [38.33, 53.63, 31.20, 41.27],
    [91.10, 120.78, 99.22, 70.62],
    [14.43, 29.76, 16.94, 11.89]
])

model2 = np.array([
    [59.34, 63.55, 63.96, 64.78],
    [50.13, 56.11, 45.29, 44.90],
    [93.85, 119.66, 108.17, 72.27],
    [15.87, 30.54, 20.41, 11.51]
])

metrics = ["MSE", "MAE", "PCC", "MS-SSIM"]
lakes = ["Superior", "Michigan", "Erie", "Ontario"]

# Plot settings
gap = 0.1
cell_width = 1.0
cell_height = 1.0

fig, ax = plt.subplots(figsize=(10, 6))
norm = plt.Normalize(-125, 125)
cmap = plt.cm.RdYlGn

def get_text_color(rgb):
    """Return black or white depending on luminance."""
    r, g, b = rgb[:3]
    luminance = 0.299*r + 0.587*g + 0.114*b
    return 'black' if luminance > 0.6 else 'white'

for i in range(model1.shape[0]):
    for j in range(model1.shape[1]):
        x = j * (cell_width + gap)
        y = i * (cell_height + gap)

        # Background colors
        color1 = cmap(norm(model1[i, j]))
        color2 = cmap(norm(model2[i, j]))

        # Rectangles
        ax.add_patch(plt.Rectangle((x, y), cell_width / 2, cell_height, facecolor=color1, edgecolor='none'))
        ax.add_patch(plt.Rectangle((x + cell_width / 2, y), cell_width / 2, cell_height, facecolor=color2, edgecolor='none'))

        # Contrast colors
        text_color1 = get_text_color(colors.to_rgb(color1))
        text_color2 = get_text_color(colors.to_rgb(color2))

        # Annotate: bold (Model 1), italic (Model 2)
        ax.text(x + 0.25, y + 0.5, f"{model1[i, j]:.1f}",
                ha='center', va='center', fontsize=10, fontweight='bold', color=text_color1)
        ax.text(x + 0.75, y + 0.5, f"{model2[i, j]:.1f}",
                ha='center', va='center', fontsize=10, style='italic', color=text_color2)

# Layout
total_width = model1.shape[1] * (cell_width + gap) - gap
total_height = model1.shape[0] * (cell_height + gap) - gap
ax.set_xlim(0, total_width)
ax.set_ylim(0, total_height)

# Ticks
xtick_positions = [j * (cell_width + gap) + cell_width / 2 for j in range(len(lakes))]
ytick_positions = [i * (cell_height + gap) + cell_height / 2 for i in range(len(metrics))]
ax.set_xticks(xtick_positions)
ax.set_yticks(ytick_positions)
ax.set_xticklabels(lakes, fontsize=12)
ax.set_yticklabels(metrics, fontsize=10, rotation=90, va='center')

# Flip y-axis
ax.invert_yaxis()

# Labels and title
ax.set_title("Percent Improvement of Model Over HRRR in Various Metrics", fontsize=14)
fig.supylabel("Metric", fontsize=12, rotation=90, va='center')

# Clean style
ax.tick_params(left=False, bottom=False)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
# plt.savefig("metrics.pdf", dpi=600, bbox_inches="tight")
plt.show()
