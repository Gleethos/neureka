package st

import neureka.Tsr
import spock.lang.Specification
import st.attention.AttentionBundle


class Advanced_NN_Convergence_Spec extends Specification
{

    def 'We can run the attention head test model classes.'()
    {
        given :
            int trainingIterations = 60
            var input  = Tsr.of( -2f, -1f, 0f, 1f, 2f ).reshape( 1, 5 )
            var target = Tsr.of( 0.2f, 1f, 0f, -1f, -0.2f ).reshape( 1, 5 )
        and :
            var model = new AttentionBundle(5, 1)
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
