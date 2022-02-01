# Take a Look! #

The following 2 code snippets show the same neural network
implementation in both Neureka and Numpy!

<table>
<tr>
<th>Neureka (Groovy)</th>
<th>Numpy (Python)</th>
</tr>
<tr>
<td> 

```groovy
var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1] ])
var y = Tsr.of(Double, [[0,1,1,0]]).T()
var syn0 = (Tsr.ofRandom(Double, 3,4) - 1) * 2
var syn1 = (Tsr.ofRandom(Double, 4,1) - 1) * 2
60_000.times {
    var l1 = Tsr.of('sig(',X.matMul(syn0),')')
    var l2 = Tsr.of('sig(',l1.matMul(syn1),')')
    var l2_delta = (y - l2)*(l2*(-l2+1))
    var l1_delta = l2_delta.matMul(syn1.T()) * (l1 * (-l1+1))
    syn1 += l1.T().matMul(l2_delta)
    syn0 += X.T().matMul(l1_delta)
}
```
 
</td>
<td>

```python
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
```

</td>
</tr>
</table>