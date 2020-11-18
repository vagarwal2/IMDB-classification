import numpy as np
def vectorize_sequences(sequences,dimensions=10000):
    results=np.zeros((len(sequences),dimensions))
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1
    return results