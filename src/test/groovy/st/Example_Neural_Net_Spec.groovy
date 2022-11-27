package st

import neureka.Tsr
import spock.lang.Specification
import testutility.nns.SimpleFeedForwardNN

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


}
