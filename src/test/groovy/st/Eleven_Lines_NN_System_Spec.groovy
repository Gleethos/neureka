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
            W1.data == [-0.9115492136933212, -2.8053196189337415, -0.2518010600685076, -1.3116921909681862, -0.6594293130862794, 2.5123812135829553, -1.0514473377002078, -1.2928097640444751, 1.3971567790475048, 2.203909930839897, -0.6782812239543352, 2.738543835064307]
            W2.data == [0.1390771364808861, -1.8305543533070094, -0.7491036073708567, 1.874957973207181]
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
            W1.data == [-0.9115492136933212, -2.8053196189337415, -0.2518010600685076, -1.3116921909681862, -0.6594293130862794, 2.5123812135829553, -1.0514473377002078, -1.2928097640444751, 1.3971567790475048, 2.203909930839897, -0.6782812239543352, 2.738543835064307]
            W2.data == [0.1390771364808861, -1.8305543533070094, -0.7491036073708567, 1.874957973207181]
    }

    def 'The pseudo random number generator works as expected for the weights used in the 11 line NN example!'()
    {
        given :
            var W1 = Tsr.ofRandom(Double, 3, 4)
            var W2 = Tsr.ofRandom(Double, 4, 1)

        expect :
            W1.data == [-0.910969595136708, -1.9627469837128895, -0.048245881734580415, -0.3554745831321771, -0.6595188311162824, 1.839723209668042, -0.7864999508162774, -1.918339420402628, 1.4035229760225527, 2.245738695936844, -0.7473176166694635, 1.9016692137691462]
            W2.data == [-0.1611330910958405, -0.9350019667613545, -0.3780200880067806, 1.4951759768158595]
    }

}
