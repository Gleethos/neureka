package it

import groovy.transform.CompileDynamic
import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
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
@CompileDynamic
@Subject([Tsr])
class Eleven_Lines_NN_System_Spec extends Specification {

    private static var RESULT_W1 = [-0.9115492136933212, -2.8053196189337415, -0.25180106006850766, -1.3116921909681862, -0.6594293130862794, 2.5123812135829557, -1.0514473377002078, -1.2928097640444747, 1.3971567790475048, 2.2039099308398966, -0.6782812239543351, 2.7385438350643065]
    private static var RESULT_W2 = [0.13907713648088624, -1.8305543533070094, -0.7491036073708569, 1.8749579732071808]

    private static var RESULT_W1_F32 = [-0.91154915, -2.8053195, -0.25180107, -1.3116925, -0.6594293, 2.5123818, -1.0514474, -1.2928098, 1.3971564, 2.2039096, -0.67828107, 2.738544]
    private static var RESULT_W2_F32 = [0.13907686, -1.8305542, -0.7491039, 1.8749583]


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
            W1.mut.data.ref == RESULT_W1
            W2.mut.data.ref == RESULT_W2
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
            W1.mut.data.ref == RESULT_W1
            W2.mut.data.ref == RESULT_W2
    }


    def 'One can write a simple float based neural network in less than 11 lines of java like code!'()
    {
        given :
            var X = Tsr.ofFloats().withShape(4,3).andFill(0f, 0f, 1f,  0f, 1f, 1f,  1f, 0f, 1f,  1f, 1f, 1f);
            var y = Tsr.ofFloats().withShape(1,4).andFill(0f, 1f, 1f, 0f).T();
            var W1 = Tsr.ofRandom(Float.class, 3,4).setRqsGradient(true);
            var W2 = Tsr.ofRandom(Float.class, 4,1).setRqsGradient(true);
            for ( int i = 0; i < 60; i++ ) {
                Tsr<Float> l2 = Tsr.of('sig(',Tsr.of('sig(',X.matMul(W1),')').matMul(W2),')');
                l2.backward(y.minus(l2));
                W1.applyGradient(); W2.applyGradient();
            }

        expect :
            W1.mut.data.ref == RESULT_W1_F32 as float[]
            W2.mut.data.ref == RESULT_W2_F32 as float[]
    }


    def 'One can write a simple double based neural network in less than 11 lines of java like code using the "@" operator!'()
    {
        given :
            var X  = Tsr.ofDoubles().withShape(4,3).andFill(0d, 0d, 1d, 0d, 1d, 1d, 1d, 0d, 1d, 1d, 1d, 1d);
            var y  = Tsr.ofDoubles().withShape(1,4).andFill(0d, 1d, 1d, 0d).T();
            var W1 = Tsr.ofRandom(Double.class, 3, 4).setRqsGradient(true);
            var W2 = Tsr.ofRandom(Double.class, 4, 1).setRqsGradient(true);
            for ( int i = 0; i < 60; i++ ) {
                var l2 = Tsr.of("sig(",Tsr.of("sig(",X,'@' as char,W1,")"),"@",W2,")");
                l2.backward(Tsr.of(y,"-",l2)); // Back-propagating the error!
                W1.applyGradient(); W2.applyGradient();
            }

        expect :
            W1.mut.data.ref == RESULT_W1
            W2.mut.data.ref == RESULT_W2
    }


    def 'The pseudo random number generator works as expected for the weights used in the 11 line NN examples!'()
    {
        given :
            var W1 = Tsr.ofRandom(Double, 3, 4)
            var W2 = Tsr.ofRandom(Double, 4, 1)

        expect :
            W1.mut.data.ref == [-0.910969595136708, -1.9627469837128895, -0.048245881734580415, -0.3554745831321771, -0.6595188311162824, 1.839723209668042, -0.7864999508162774, -1.918339420402628, 1.4035229760225527, 2.245738695936844, -0.7473176166694635, 1.9016692137691462]
            W2.mut.data.ref == [-0.1611330910958405, -0.9350019667613545, -0.3780200880067806, 1.4951759768158595]
    }

}
