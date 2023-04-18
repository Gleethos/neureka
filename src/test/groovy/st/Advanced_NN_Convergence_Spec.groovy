package st

import neureka.Tsr
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title
import st.attention.QuasiMultiHeadAttention

@Title("Training a Neural Network Class")
@Narrative('''

    When designing larger neural network architectures, what you would usually do is
    to create a class that represents the whole model (which itself might be composed
    of smaller models). 
    
    This class would then represent something that can be executed and then trained.
    This Specification shows how to do this using the `QuasiMultiHeadAttention` model,
    which is an example of a model that is composed of smaller models based on the
    attention architecture, as can be found in the popular transformer model architecture.

''')
class Advanced_NN_Convergence_Spec extends Specification
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
            var input  = Tsr.of( -2f, -1f, 0f, 1f, 2f ).reshape( 1, 5 )
            var target = Tsr.of( 0.2f, 1f, 0f, -1f, -0.2f ).reshape( 1, 5 )
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

}
