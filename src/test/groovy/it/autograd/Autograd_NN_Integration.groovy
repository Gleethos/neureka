package it.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import org.junit.Ignore;
import spock.lang.Specification;

class Autograd_NN_Integration extends Specification
{

/*
    def 'Neural Network'(){

        given :
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested( false )
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed( false )
            Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp( false )
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
                def predictions = [:]
                100.times {
                    def pred = forward(X)
                    predictions[it] = pred.toString()
                    def error = (y - pred)
                    //def loss = (error**2).mean()
                    println(error)
                    pred.backward(error)
                }

            then :
                predictions[0].contains("(3):[0.40330E0, 0.43917E0, 0.44033E0]") // This has been checked thoroughly!
    }


    def 'Autograd can be applied to simple linear regression'()
    {
        given :
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested( false )
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed( false )
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp( false )
        def X = new Tsr(
                [[0.6667, 1.0000],
                 [0.3333, 0.5556],
                 [1.0000, 0.6667]]
        ).T()
        def y = new Tsr(
                [[0.9200],
                 [1.0000]]
        )
        def sig = Function.create("sig(I[0])")
        //def W1 = new Tsr(
        //        [[-1.1843,  0.0146, -1.4647],
        //         [-1.4020, -1.0129,  0.6256]]
        //).setRqsGradient(true)
        def W1 = new Tsr(
                [[ 1.8095],
                 [-0.4269],
                 [-1.1110]]
        ).setRqsGradient(true)

        def forward = ( Tsr x ) -> {
            def z = x.dot(W1)
            def o = z//sig(z)
            return o
        }

        when :
        def predictions = [:]
        100.times {
            def pred = forward(X)
            predictions[it] = pred.toString()
            def error = (y - pred)
            //def loss = (error**2).mean()
            println(error)
            pred.backward(error)
        }

        then :
            true
    }

*/
}
