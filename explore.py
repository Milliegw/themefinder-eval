import random
from src.metrics import f1
from collections import Counter

y_true = [1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 1, 0, 1, 0]

rng = random.Random(1)
n = len(y_true)

picks = rng.choices(range(n), k=n)
print("picked indices:", picks)

new_true = [y_true[i] for i in picks]
new_pred = [y_pred[i] for i in picks]
print("new y_true:", new_true)
print("new y_pred:", new_pred)
print("F1 on this sample:", f1(new_true, new_pred))

# Step 2: 1,000 pretend samples
scores = []
for _ in range(1000):
    picks = rng.choices(range(n), k=n)
    new_true = [y_true[i] for i in picks]
    new_pred = [y_pred[i] for i in picks]
    scores.append(f1(new_true, new_pred))

print("original F1:", f1(y_true, y_pred))
print("lowest:", min(scores), "highest:", max(scores))
print("most common scores:", Counter(round(s, 2) for s in scores).most_common(5))

# Step 3: the middle 95%
scores.sort()
print("95% interval:", scores[25], "to", scores[974])