package st

import neureka.Shape
import neureka.Tensor
import neureka.ndim.Filler
import neureka.optimization.Optimizer
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Examining Neural Network Architecture Snippets")
@Narrative('''

    This specification is intended to showcase some basic building blocks of 
    various neural network architectures.

''')
class NN_Concepts_Spec extends Specification
{

    def 'The attention mechanism (found in the commonly known transformer) demonstrated.'()
    {
        reportInfo """
            The attention mechanism is a core component of the transformer architecture and
            most likely the reason why it is so successful in natural language processing.
            
            Here you can see that the query and key weight matrices are trained
            if there is only one input vector.
        """
        given : 'We define an input containing a single vector and target data.'
            var x1  = Tensor.of( -2f, -1f ).reshape( 1, 2 )
            var y = Tensor.of( 1.2f, -0.77f ).reshape( 1, 2 )
        and : 'We define a custom weight matrix filler lambda.'
            Filler<Float> filler = ( int i, int[] idx ) -> (float) (Math.abs( Math.pow( 31, 42 + i ) % 11 ) - 5) / 2f
        and :
            var Wk = Tensor.of(Float.class, Shape.of(2, 2), filler).setRqsGradient(true)
            var Wq = Tensor.of(Float.class, Shape.of(2, 2), filler).setRqsGradient(true)
            var Wv = Tensor.of(Float.class, Shape.of(2, 2), filler).setRqsGradient(true)

            Wk.set(Optimizer.ADAM)
            Wq.set(Optimizer.ADAM)
            Wv.set(Optimizer.ADAM)

        and : 'Finally define everything for training.'
            int trainingIterations = 10
            var pred = null
            var loss = []
            var trainer = { x ->
                Tensor<Float> key   = x.matMul(Wk)
                Tensor<Float> query = x.matMul(Wq)
                Tensor<Float> value = x.matMul(Wv)

                var attention = query.matMul(key.T()).softmax(1)

                pred = attention.matMul(value)

                var error = ( ( y - pred ) ** 2f ).sum()
                error.backward()

                // Applying gradients:
                Wk.applyGradient()
                Wq.applyGradient()
                Wv.applyGradient()

                return error.item()
            }

        when :
            trainingIterations.times {
                loss << trainer(x1)
            }
        then :
            pred.shape == [1, 2]
            loss.size() == trainingIterations
            loss[0] > loss[trainingIterations-1]
            loss[0].round(3) == 0.633f
            loss[trainingIterations-1].round(3) == 0.255f
    }

}
