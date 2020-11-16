import re
import sys
from pyspark import SparkConf, SparkContext
conf = SparkConf()
sc = SparkContext(conf=conf)
lines = sc.textFile(sys.argv[1])
words = lines.flatMap(lambda l:re.split(r'[^\w]+',l))
bigrams=words.flatMap(lambda x:list((x[i],x[i+1]) for i in range(0,len(x)-1)))
pairs = bigrams.map(lambda w: (w, 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
counts.saveAsTextFile(sys.argv[2])
sc.stop()

