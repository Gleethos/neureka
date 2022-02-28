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
var y = Tsr.of(Double, [[0,1,1,0]]).T
var W1 = Tsr.ofRandom(Double, 3,4)
var W2 = Tsr.ofRandom(Double, 4,1)
60_000.times {
    var l1 = Tsr.of('sig(',X.dot(W1),')')
    var l2 = Tsr.of('sig(',l1.dot(W2),')')
    var l2_delta = (y - l2)*(l2*(-l2+1))
    var l1_delta = l2_delta.dot(W2.T) * (l1 * (-l1+1))
    W2 += l1.T.dot(l2_delta)
    W1 += X.T.dot(l1_delta)
}
```
 
</td>
<td>

```python
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
W1 = 2*np.random.random((3,4)) - 1
W2 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,W1))))
    l2 = 1/(1+np.exp(-(np.dot(l1,W2))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(W2.T) * (l1 * (1-l1))
    W2 += l1.T.dot(l2_delta)
    W1 += X.T.dot(l1_delta)
```

</td>
</tr>
</table>

---

## But wait! It gets even better! ##

All of this `l1_delta` and `l2_delta` stuff looks really
complicated and is really distracting from the actual architecture
of the network... That is why Neureka has this part
of building a neural network automated for you!
You simply specify which tensors require gradients,
and then you send the error values through the `backward`
method to your weights (and their gradients).
After each pass you tell your weights to apply their gradients.
Done!

<table>
<tr>
<th>Neureka (Groovy)</th>
<th>Numpy (Python)</th>
</tr>
<tr>
<td> 

```groovy
var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
var y = Tsr.of(Double, [[0, 1, 1, 0]]).T
var W1 = Tsr.ofRandom(Double, 3, 4).setRqsGradient(true)
var W2 = Tsr.ofRandom(Double, 4, 1).setRqsGradient(true)
60_000.times {
    var l2 = Tsr.of('sig(',Tsr.of('sig(',X.dot(W1),')').dot(W2),')')
    l2.backward(y - l2) // Back-propagating the error!
    W1.applyGradient(); W2.applyGradient()
}
```

</td>
<td>

```python
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
W1 = 2*np.random.random((3,4)) - 1
W2 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,W1))))
    l2 = 1/(1+np.exp(-(np.dot(l1,W2))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(W2.T) * (l1 * (1-l1))
    W2 += l1.T.dot(l2_delta)
    W1 += X.T.dot(l1_delta)
```

</td>
</tr>
</table>
