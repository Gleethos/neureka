package st


import neureka.Tensor
import neureka.optimization.Optimizer
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title
import st.attention.QuasiMultiHeadAttention
import testutility.nns.SimpleFeedForwardNN

import java.util.function.Consumer

@Title("Training a Neural Network Class")
@Narrative('''

    When designing larger neural network architectures, what you would usually do is
    to create a class that represents the whole model (which itself might be composed
    of smaller models). 
    
    This class would then represent something that can be executed and then trained.
    This Specification shows how to instantiate, execute and train various 
    pre-defined example neural network models.
    
''')
class Training_NNs_Spec extends Specification
{

    def 'We can run the attention head test model.'()
    {
        reportInfo """
            This little test simply executes the `QuasiMultiHeadAttention` model class
            and checks if the loss is decreasing over time.
            You can check out how this is implemented in the `QuasiMultiHeadAttention` class.
            Here you will only see how the training is executed.
        """
        given :
            int trainingIterations = 60
            var input  = Tensor.of( -2f, -1f, 0f, 1f, 2f ).reshape( 1, 5 )
            var target = Tensor.of( 0.2f, 1f, 0f, -1f, -0.2f ).reshape( 1, 5 )
        and :
            var model = new QuasiMultiHeadAttention(5, 1)
        when :
            var pred = null
            var loss = []
            trainingIterations.times {
                pred = model.run( [input] )
                loss << model.train( target )
                //println( "Loss: ${loss[loss.size()-1]} Prediction: ${pred.items()}" )
            }
        then :
            pred.shape == [1, 5]
            loss.size() == trainingIterations
            loss[0] > loss[trainingIterations-1]
            loss[0].round(3) == 9.127
            loss[trainingIterations-1] < 0.75
    }

    def 'A simple 3 layer neural network converges.'()
    {
        given :
            var predictor = new SimpleFeedForwardNN(5, 42)
            var input = Tensor.of( -0.2f, -0.1f, 0f, 0.1f, 0.2f ).reshape( 1, 5 )
            var target = Tensor.of( 0.2f, 0.1f, 0f, -0.1f, -0.2f ).reshape( 1, 5 )
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

    def 'A very simple 1 layer NN converges.'( Consumer<Tensor<Float>> applyOptimizer )
    {
        given :
            var inputs = Tensor.ofFloats().withShape( 2, 6 ).andFill(-4f..3f)
            var weights = Tensor.ofRandom(Float, 6, 1)
            var targets = Tensor.of( 0.2f, -0.1f, 0.5f, 1.2f, -0.3f, 0.2f ).reshape( 2, 1 )
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
