import numpy as np 
import matplotlib.pyplot as plt

results = []

for seed in range(13420):
    np.random.seed(seed)
    
    score = np.random.normal(loc = 32,scale = 4)
    results.append(score)
results = np.array(results)

print("Results:", results)
print("Mean:", np.mean(results))
print("Variance:", np.var(results))
print("Standard deviation:", np.std(results))


plt.plot(range(1,len(results)+1),results,marker = 'o')


plt.xlabel('Experiments')
plt.ylabel('Scores')
plt.title("Scores Across Random Seeds")
plt.grid()
plt.show()