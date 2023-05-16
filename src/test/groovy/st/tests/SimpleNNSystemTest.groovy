package st.tests

import neureka.Tsr
import neureka.devices.Device
import testutility.mock.DummyDevice

class SimpleNNSystemTest
{
    enum Mode {
        CONVOLUTION,
        MAT_MUL
    }

    private final Mode mode
    private final def _$
    private final def back
    private final def back2

    SimpleNNSystemTest(Mode mode) {
        this.mode = mode
        switch (mode) {
            case Mode.CONVOLUTION:
                _$   = { a, b -> Tsr.of('I[0] x I[1]', [a, b]) }
                back = { a, b -> Tsr.of('I[0] x I[1]', [a, b]) }
                back2 = { a, b -> Tsr.of('I[0] x I[1]', [a, b]) }; break
            case Mode.MAT_MUL:
                _$    = { a, b -> a.matMul(b) }
                back  = { a, b -> a.T().matMul(b) }
                back2 = { a, b -> a.matMul(b.T()) }; break

            default: _$ = null; back = null
        }
    }

    void on(Device device)
    {
        boolean doMatMul = ( mode == Mode.MAT_MUL )

        var inputShape =  ( doMatMul ? [5,3] : [5, 3, 1]             )// (5x3)
        var w1Shape =     ( doMatMul ? [3,4] : [1, inputShape[1], 4] )// (3x4)
        var w2Shape =     ( doMatMul ? [4,1] : [1, 1, 4]             )// (4x1)
        var outputShape = ( doMatMul ? [5,1] : [5, 1, 1]             )// (5x1)
        var layer1Shape = ( doMatMul ? [5,4] : [5, 1, 4]             )// (5x4)
        /*
                (5x3)|(3x4)->(4x1)|(5x1)
                (5x4)*(4x3)<-(5x1)*(1x4)
         */

        var X = Tsr.of( // input data: 5 vectors in binary form
                                inputShape, // (5x3)
                                [
                                    0d, 0d, 1d,
                                    1d, 1d, 0d,
                                    1d, 0d, 1d,
                                    0d, 1d, 1d,
                                    1d, 1d, 1d
                                ]
                            ).to(device)

        // output values (labels)
        var y = Tsr.of(outputShape,[0d,1d,1d,1d,0d]).to(device)

        var input = X
        var weights1 = Tsr.of(w1Shape,
                                [4.17022005e-01d, 7.20324493e-01d, 1.14374817e-04d, 3.02332573e-01d,
                                 1.46755891e-01d, 9.23385948e-02d, 1.86260211e-01d, 3.45560727e-01d,
                                 3.96767474e-01d, 5.38816734e-01d, 4.19194514e-01d, 6.85219500e-01d]
                            ).to(device)
        /*
            [1x5x4]:(...)
            w1 (3, 4) :
            [[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
             [1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01]
             [3.96767474e-01 5.38816734e-01 4.19194514e-01 6.85219500e-01]]
        */
        var weights2 = Tsr.of(w2Shape, [0.20445225d, 0.87811744d, 0.02738759d, 0.67046751d]).to(device)
        /*
            [1x1x4]:...
            w2 (4, 1) :
            [[0.20445225]
             [0.87811744]
             [0.02738759]
             [0.67046751]]
         */
        var output = Tsr.of(outputShape, [0d, 0d, 0d, 0d, 0d]).to(device)
        /*
            [5x1x1]...
            out (5, 1) :
            [[0.0]
             [0.0]
             [0.0]
             [0.0]
             [0.0]]
        */
        var layer1 = Tsr.of(layer1Shape, 0d).to(device)

        // iterate 500 times
        for ( i in 0..500 ){
            feedforward(weights1, weights2, input, output, layer1)
            backprop(weights1, weights2, input, output, layer1, y)
        }

        assert output.getItemsAs( double[].class )[0] >= 0.0 && output.getItemsAs( double[].class )[0] <= 1.0
        assert output.getItemsAs( double[].class )[1] >= 0.0 && output.getItemsAs( double[].class )[1] <= 1.0
        assert output.getItemsAs( double[].class )[2] >= 0.0 && output.getItemsAs( double[].class )[2] <= 1.0

        assert output.getItemsAs( double[].class )[0] >= 0.0 &&output.getItemsAs( double[].class )[0] <= 0.1
        assert output.getItemsAs( double[].class )[1] >= 0.95 &&output.getItemsAs( double[].class )[1] <= 1.0
        assert output.getItemsAs( double[].class )[2] >= 0.95 &&output.getItemsAs( double[].class )[2] <= 1.0
        assert output.getItemsAs( double[].class )[3] >= 0.95 &&output.getItemsAs( double[].class )[3] <= 1.0
        assert output.getItemsAs( double[].class )[4] >= 0.0 &&output.getItemsAs( double[].class )[4] <= 0.1

        assert output.getItemsAs( double[].class )[0] >= 0.0 && output.getItemsAs( double[].class )[0] <= 0.0055
        assert output.getItemsAs( double[].class )[1] >= 0.95 && output.getItemsAs( double[].class )[1] <= 1.0
        assert output.getItemsAs( double[].class )[2] >= 0.95 && output.getItemsAs( double[].class )[2]<= 1.0
        assert output.getItemsAs( double[].class )[3] >= 0.95 && output.getItemsAs( double[].class )[3] <= 1.0
        assert output.getItemsAs( double[].class )[4] >= 0.05 && output.getItemsAs( double[].class )[4] <= 0.06

        if ( device instanceof DummyDevice ) {
            // When we are not running on the GPU we can assert the result almost deterministically
            var asString = output.items.collect({it.toString().substring(0,13)}).join(',')

            assert asString == '0.00513865223,0.96433016244,0.97344395635,0.96021058419,0.05098795507'
        }
    }

    static Tsr sigmoid(Tsr x) {
        return Tsr.of("sig(I[0])", x)
        // Other possibilities:
        //return Tsr.of(((Tsr.Create.E(x.shape())**(-x))+1), "1/I[0]")
        //return 1.0 / (1 + Tsr.Create.E(x.shape())**(-x))
    }

    static Tsr sigmoid_derivative(Tsr x) {
        return x * (-x + 1)
    }

    void feedforward(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1) {
        var in0 = _$(input, weights1)
        layer1.mut[] = sigmoid(in0)
        //println(layer1.toString("shp")+"=sig(  I"+input.toString("shp")+" X W"+weights1.toString("shp")+" )")
        var in1 = _$(layer1, weights2)
        output.mut[] = sigmoid(in1)
        //println(output.toString("shp")+"=sig( L1"+layer1.toString("shp")+" X W"+weights2.toString("shp")+" )\n")
    }

    void backprop(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1, Tsr y) {
        // application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        Tsr delta = (y - output) * 2
        Tsr derivative = delta * 2 * sigmoid_derivative(output)
        Tsr d_weights2 = back(layer1, derivative)
        Tsr temp = back2(derivative, weights2)
        Tsr d_weights1 = back(input, temp * sigmoid_derivative(layer1))
        // update the weights with the derivative (slope) of the loss function
        weights1.mut[] = weights1 + d_weights1
        weights2.mut[] = weights2 + d_weights2
    }


}
