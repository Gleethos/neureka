package st.tests


import neureka.Tsr
import neureka.devices.Device

class SimpleNNSystemTest
{
    static void on(Device device)
    {
        Tsr X = Tsr.of(// input data: 5 vectors in binary form
                [5, 3, 1],
                [
                        0, 0, 1,
                        1, 1, 0,
                        1, 0, 1,
                        0, 1, 1,
                        1, 1, 1
                ]
        ).set(device)

        Tsr y = Tsr.of(// output values (labels)
                [5, 1, 1],[0,1,1,1,0]
        ).set(device)// [1, 5, 1],(0,1,1,1,0)

        Tsr input = X
        Tsr weights1 = Tsr.of([1, input.shape()[1], 4],
                [4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01,
                 1.46755891e-01, 9.23385948e-02, 1.86260211e-01, 3.45560727e-01,
                 3.96767474e-01, 5.38816734e-01, 4.19194514e-01, 6.85219500e-01]
        ).set(device)
        /*
            [1x5x4]:(...)
            w1 (3, 4) :
            [[4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
             [1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01]
             [3.96767474e-01 5.38816734e-01 4.19194514e-01 6.85219500e-01]]
        */
        Tsr weights2 = Tsr.of([1, 1, 4], [0.20445225, 0.87811744, 0.02738759, 0.67046751]).set(device)
        /*
            [1x1x4]:...
            w2 (4, 1) :
            [[0.20445225]
             [0.87811744]
             [0.02738759]
             [0.67046751]]
         */
        Tsr output = Tsr.of(y.shape(), [0.0, 0.0, 0.0, 0.0, 0.0]).set(device)
        /*
            out (5, 1) :
            [[0.0]
             [0.0]
             [0.0]
             [0.0]
             [0.0]]
        */
        Tsr layer1 = Tsr.of()
        /*
              inp (5, 3) :
              [[0, 0, 1]
               [1, 1, 0]
               [1, 0, 1]
               [0, 1, 1]
               [1, 1, 1]]
         */

        // iterate 500 times
        for( i in 0..500){
            feedforward(weights1, weights2, input, output, layer1)
            backprop(weights1, weights2, input, output, layer1, y)
        }

        assert output.value64()[0] >= 0.0 && output.value64()[0] <= 1.0
        assert output.value64()[1] >= 0.0 && output.value64()[1] <= 1.0
        assert output.value64()[2] >= 0.0 && output.value64()[2] <= 1.0

        assert output.value64()[0] >= 0.0 &&output.value64()[0] <= 0.1
        assert output.value64()[1] >= 0.95 &&output.value64()[1] <= 1.0
        assert output.value64()[2] >= 0.95 &&output.value64()[2] <= 1.0
        assert output.value64()[3] >= 0.95 &&output.value64()[3] <= 1.0
        assert output.value64()[4] >= 0.0 &&output.value64()[4] <= 0.1

        assert output.value64()[0] >= 0.0 && output.value64()[0] <= 0.0055
        assert output.value64()[1] >= 0.95 && output.value64()[1] <= 1.0
        assert output.value64()[2] >= 0.95 && output.value64()[2]<= 1.0
        assert output.value64()[3] >= 0.95 && output.value64()[3] <= 1.0
        assert output.value64()[4] >= 0.05 && output.value64()[4] <= 0.06

    }

    static Tsr sigmoid(Tsr x) {
        return Tsr.of(x, "sig(I[0])")
        //return Tsr.of(((Tsr.Create.E(x.shape())**(-x))+1), "1/I[0]")
        //return 1.0 / (1 + Tsr.Create.E(x.shape())**(-x))
    }

    static Tsr sigmoid_derivative(Tsr x) {
        return x * (-x + 1)
    }

    static void feedforward(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1) {
        Tsr in0 = Tsr.of([input, weights1], "i0xi1")
        layer1[] = sigmoid(in0)
        //println(layer1.toString("shp")+"=sig(  I"+input.toString("shp")+" X W"+weights1.toString("shp")+" )")
        Tsr in1 = Tsr.of([layer1, weights2], "i0xi1")
        output[] = sigmoid(in1)
        //println(output.toString("shp")+"=sig( L1"+layer1.toString("shp")+" X W"+weights2.toString("shp")+" )\n")
    }

    static void backprop(Tsr weights1, Tsr weights2, Tsr input, Tsr output, Tsr layer1, Tsr y) {
        // application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        Tsr delta = (y - output)*2
        Tsr derivative = delta*2*sigmoid_derivative(output)
        Tsr d_weights2 = Tsr.of(
                [layer1, (derivative)],
                "i0xi1"
        )
        Tsr d_weights1 = Tsr.of(
                [input, (Tsr.of([derivative, weights2], "i0xi1") * sigmoid_derivative(layer1))],
                "i0xi1"
        )
        // update the weights with the derivative (slope) of the loss function
        weights1[] = weights1 + d_weights1
        weights2[] = weights2 + d_weights2
    }


}
