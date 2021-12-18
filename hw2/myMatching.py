import numpy as np

class myMatching():
    def __init__(self, queryIdx=None, trainIdx=None, distance=None):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def myknn(des1, des2, num):
    matches = []
    for i, f in enumerate(des1):
        fmatch = []
        f_rep = f[None,:].repeat(des2.shape[0], 0)
        distance = np.linalg.norm(f_rep - des2, axis=-1)
        idx = np.argsort(distance)
        for n in range(num):
            fmatch.append(myMatching(i, idx[n], distance[idx[n]]))
        matches.append(fmatch)
    return matches
