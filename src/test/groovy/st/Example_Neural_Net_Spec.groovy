package st

import neureka.Tsr
import neureka.optimization.Optimizer
import neureka.optimization.implementations.ADAM
import neureka.optimization.implementations.RMSProp
import neureka.optimization.implementations.SGD
import spock.lang.Specification
import testutility.nns.SimpleFeedForwardNN

import java.util.function.Consumer

class Example_Neural_Net_Spec extends Specification
{
    def 'A simple 3 layer neural network converges.'()
    {
        given :
            var predictor = new SimpleFeedForwardNN(5, 42)
            var input = Tsr.of( -0.2f, -0.1f, 0f, 0.1f, 0.2f ).withShape( 1, 5 )
            var target = Tsr.of( 0.2f, 0.1f, 0f, -0.1f, -0.2f ).withShape( 1, 5 )
        when :
            var pred
            var loss = []
            100.times {
                pred = predictor.forward( input )
                loss << predictor.train( target )
                //println( "Loss: ${loss.last()}" )
                //println( "Prediction: ${pred}" )
            }
        then :
            pred.shape == [1, 5]
            loss.size() == 100
            loss[0] > loss[99]
            loss[0] > 1
            loss[99] < 0.005
    }

    def 'A very simple 1 layer NN converges.'( Consumer<Tsr<Float>> applyOptimizer )
    {
        given :
            var inputs = Tsr.ofFloats().withShape( 2, 6 ).andFill(-4f..3f)
            var weights = Tsr.ofRandom(Float, 6, 1)
            var targets = Tsr.of( 0.2f, -0.1f, 0.5f, 1.2f, -0.3f, 0.2f ).withShape( 2, 1 )
        and :
            weights.setRqsGradient( true )
            applyOptimizer.accept(weights)
        and :
            var pred
            var losses = []
        when :
            100.times {
                pred = inputs.matMul( weights ).tanh()
                var loss = ((pred - targets)**2).sum()
                loss.backward()
                weights.applyGradient()
                losses << loss.item()
            }
        then :
            pred.shape == [2, 1]
            losses[0] > losses[losses.size()-1]
            losses[0] > 2
            losses[losses.size()-1] < 0.5

        where :
            applyOptimizer << [
                    { it.set(Optimizer.SGD.withLearningRate(0.03)) },
                    { it.set(Optimizer.ADAM) },
                    { it.set(Optimizer.RMSProp.withLearningRate(0.05)) }
                ]
    }

}
