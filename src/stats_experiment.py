import numpy as np

results = []

for seed in range(10):
    np.random.seed(533)

    score = np.random.normal(loc=80, scale=2)

    results.append(score)

results = np.array(results)

print("Results:", results)
print("Mean:", np.mean(results))
print("Variance:", np.var(results))
print("Standard deviation:", np.std(results))