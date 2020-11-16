from pyspark import SparkConf, SparkContext
import re
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
conf = SparkConf()
sc = SparkContext(conf=conf)

MAX_ITER = 20
k = 10


# Load data
def load(path):
    lines = sc.textFile(path)
    slines = lines.map(lambda line: [float(v) for v in re.split(r' ', line)])
    return slines

# Distance
def dist(l, c, dist_type):
    if dist_type == 'Euclidean':
        return np.linalg.norm(np.array(l)-c, 2)
    else:
        raise ValueError("Invalid Distance Type!")

# Minimum and corresponding index
def min_and_ind(l):
    return (np.min(l), np.argmin(l))

# Calculate L2 centroids
def c_2(l):
    c = np.zeros(len(l[0]))
    for v in l:
        c += np.array(v)
    return c/len(l)
    

# Do k-means clustering
def clustering(data, c, dist_type):
    
    it = 0
    costs = []
    c_array = np.array(c.collect())
    
    while it < MAX_ITER:
        # print(c_array[0][0:3])
        clusters = data.map(lambda line: (line, min_and_ind([dist(line, cc, dist_type) for cc in c_array])))\
			.map(lambda line: (line[1][1], (line[0], line[1][0])))\
			.groupByKey()\
			.mapValues(lambda vs: [v for v in vs])
        cost = clusters.map(lambda line: (1, sum([v[1] for v in line[1]])))\
                       .reduceByKey(lambda n1, n2: n1+n2)\
                       .collect()[0][1]
        centroids = clusters.map(lambda line: [v[0] for v in line[1]])\
                            .map(lambda line: list(c_2(line)))\
                            .collect()

        it += 1
        costs.append(cost)
        c_array = np.array(centroids)
    
    return costs

# Plot the cost versus iteration
def solve(data, far, near, dist_type):
    
    costs1 = clustering(data, far, dist_type)
    costs2 = clustering(data, near, dist_type)
    
    plt.figure(figsize = (17,5))
    plt.subplot(121)
    plt.plot(costs1, 'ro-')
    plt.title('The percent change after 10 iterations for far is '+str((1-costs1[9]/costs1[0])*100)+'%')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.subplot(122)
    plt.plot(costs2, 'bo-')     
    plt.title('The percent change after 10 iterations for near is '+str((1-costs2[9]/costs2[0])*100)+'%')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.savefig(dist_type+'.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    data = load(sys.argv[1])
    far = load(sys.argv[2])
    near = load(sys.argv[3])
    solve(data, far, near, 'Euclidean')

