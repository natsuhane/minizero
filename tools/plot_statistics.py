import pandas as pd
import matplotlib.pyplot as plt

# -------- num negative board histogram --------
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("statistic/value_histogram_B.csv")
x = (df["bin_left"] + df["bin_right"]) / 2

# Plot each threshold as a line
plt.figure(figsize=(10, 6))
for i in range(1, 11):
    col = f"thresh{i:02d}"
    plt.plot(x, df[col], marker='o', label=col)

plt.xlabel("True Board Value")
plt.ylabel("Negative Count")
plt.title("Num Negative Boards vs True Board Value (Black's Turn, bin = 0.2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("statistic/value_histogram_B.png", dpi=150)
plt.close()

df = pd.read_csv("statistic/value_histogram_W.csv")
x = (df["bin_left"] + df["bin_right"]) / 2

# Plot each threshold as a line
plt.figure(figsize=(10, 6))
for i in range(1, 11):
    col = f"thresh{i:02d}"
    plt.plot(x, df[col], marker='o', label=col)

plt.xlabel("True Board Value")
plt.ylabel("Negative Count")
plt.title("Num Negative Boards vs True Board Value (White's Turn, bin = 0.2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("statistic/value_histogram_W.png", dpi=150)
plt.close()

# -------- value difference histogram --------
df = pd.read_csv("statistic/value_diff_histogram.csv")
# Compute bin center as x-axis
x = (df["bin_left"] + df["bin_right"]) / 2

plt.figure(figsize=(10, 6))

# Plot black and white lines
plt.plot(x, df["black"], marker='o', label="black's turn")
plt.plot(x, df["white"], marker='o', label="white's turn")

plt.xlabel("Value difference")
plt.ylabel("Count")
plt.title("Value Difference (Neg - Pos) for all Negatives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("statistic/value_diff_histogram.png", dpi=150)
plt.close()

# -------- num positive board histogram --------
df = pd.read_csv("statistic/pos_histogram.csv")
x = (df["bin_left"] + df["bin_right"]) / 2
plt.figure(figsize=(10, 6))

# Plot black and white lines
plt.plot(x, df["black"], marker='o', label="black's turn")
plt.plot(x, df["white"], marker='o', label="white's turn")

plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Statistics for all Positives")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("statistic/pos_histogram.png", dpi=150)
plt.close()
