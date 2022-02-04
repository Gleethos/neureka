package st

import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("NN Code Golfing!")
@Narrative('''

    This system test specification uses the following Numpy
    code as reference implementation for the equivalent in Neureka
    or similar implementations and variations.
    The code below is a simple neural network in only 11 lines of code.

    ´´´
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
    ´´´

''')
class Eleven_Lines_NN_System_Spec extends Specification {

    def 'One can write a simple neural network with custom back-prop in 11 lines of code!'()
    {
        given :
            var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
            var y = Tsr.of(Double, [[0, 1, 1, 0]]).T
            var W1 = Tsr.ofRandom(Double, 3, 4)
            var W2 = Tsr.ofRandom(Double, 4, 1)
            60.times {
                var l1 = Tsr.of('sig(', X.matMul(W1), ')')
                var l2 = Tsr.of('sig(', l1.matMul(W2), ')')
                var l2_delta = (y - l2) * (l2 * (-l2 + 1))
                var l1_delta = l2_delta.matMul(W2.T) * (l1 * (-l1 + 1))
                W2 += l1.T.matMul(l2_delta)
                W1 += X.T.matMul(l1_delta)
            }

        expect :
            W1.data == [-0.30756184006769133, -0.3921063814502533, -3.3119082858230864, 0.3207614185493358, -0.21927511078226772, -0.38713843348435945, -3.0818022111884, 0.2563838127486927, 1.8087059731824258, 0.9805434973788656, 0.43573786374580203, -2.2406401775280735]
            W2.data == [0.32613873897767304, 0.4543018266007193, -2.889923160670668, 0.3354624358387805]
    }

    def 'One can write a simple neural network in less than 11 lines of code!'()
    {
        given :
            var X = Tsr.of(Double, [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
            var y = Tsr.of(Double, [[0, 1, 1, 0]]).T
            var W1 = Tsr.ofRandom(Double, 3, 4).setRqsGradient(true)
            var W2 = Tsr.ofRandom(Double, 4, 1).setRqsGradient(true)
            60.times {
                var l2 = Tsr.of('sig(',Tsr.of('sig(',X.matMul(W1),')').matMul(W2),')')
                l2.backward(y - l2)
                W1.applyGradient(); W2.applyGradient()
            }

        expect :
            W1.data == [-0.30756184006769133, -0.3921063814502533, -3.3119082858230864, 0.3207614185493358, -0.21927511078226772, -0.38713843348435945, -3.0818022111884, 0.2563838127486927, 1.8087059731824258, 0.9805434973788656, 0.43573786374580203, -2.2406401775280735]
            W2.data == [0.32613873897767304, 0.4543018266007193, -2.889923160670668, 0.3354624358387805]
    }


}
