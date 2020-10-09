package it.autograd

import neureka.Tsr
import neureka.calculus.Function
import org.junit.Ignore;
import spock.lang.Specification;

class Autograd_NN_Integration extends Specification
{

    /*
    def 'Neural Network'(){

        given :
            def X = new Tsr(
                [[0.6667, 1.0000],
                 [0.3333, 0.5556],
                 [1.0000, 0.6667]]
            )
            def y = new Tsr(
                    [[0.9200],
                     [1.0000],
                     [0.8900]]
            )
            def sig = Function.create("sig(I[0])")
            def W1 = new Tsr(
                    [[-1.1843,  0.0146, -1.4647],
                     [-1.4020, -1.0129,  0.6256]]
            ).setRqsGradient(true)
            def W2 = new Tsr(
                    [[ 1.8095],
                     [-0.4269],
                     [-1.1110]]
            ).setRqsGradient(true)

            def forward = ( Tsr x ) -> {
                def z = x.dot(W1)
                def z2 = sig(z)
                def z3 = z2.dot(W2)
                def o = sig(z3)
                return o
            }

            when :
                100.times{
                    def pred = forward(X)
                    def error = (y - pred)
                    def loss = (error**2).mean()
                    println(loss)
                    pred.backward(error)
                }

            then :
                true

    }
    */


}
