import neureka.Tsr

/*
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
*/
var version_1 = {
    var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    var y = Tsr.of(Double, [[0, 1, 1, 0]]).T()
    var W1 = (Tsr.ofRandom(Double, 3, 4) - 1) * 2
    var W2 = (Tsr.ofRandom(Double, 4, 1) - 1) * 2
    60_000.times {
        var l1 = Tsr.of('sig(', X.matMul(W1), ')')
        var l2 = Tsr.of('sig(', l1.matMul(W2), ')')
        var l2_delta = (y - l2) * (l2 * (-l2 + 1))
        var l1_delta = l2_delta.matMul(W2.T()) * (l1 * (-l1 + 1))
        W2 += l1.T().matMul(l2_delta)
        W1 += X.T().matMul(l1_delta)
    }
}

var version_2 = {
    var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    var y = Tsr.of(Double, [[0, 1, 1, 0]]).T()
    var W1 = ((Tsr.ofRandom(Double, 3, 4) - 1) * 2).setRqsGradient(true)
    var W2 = ((Tsr.ofRandom(Double, 4, 1) - 1) * 2).setRqsGradient(true)
    60_000.times { 
        var l2 = Tsr.of('sig(',Tsr.of('sig(',X.matMul(W1),')').matMul(W2),')')
        ((y - l2)**2).mean().backward()
        W1.applyGradient(); W2.applyGradient()
    }
}

version_2()