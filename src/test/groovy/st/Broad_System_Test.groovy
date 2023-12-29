package st

import neureka.Neureka
import neureka.Shape
import neureka.Tensor
import neureka.view.NDPrintSettings
import spock.lang.Specification
import st.tests.BroadSystemTest

class Broad_System_Test extends Specification
{
    def setupSpec() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'The long broad integration test runs successfully.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on() // This is the actual test.
    }

    def 'A function with expression "softplus((I[0]xI[1])*-100)" can be backpropagated.'()
    {
        given :
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true);
            Tensor<Double> tensor1 = Tensor.of(Shape.of(1, 3), 2d);
            Tensor<Double> tensor2 = Tensor.of(Double).withShape(2, 1).all(-1.0);
            tensor1.setRqsGradient(true);
            tensor2.setRqsGradient(true);

        when :
            Tensor<Double> product1 = Tensor.of("softplus((I[0]xI[1])*-100)", [tensor1, tensor2]);
            Tensor<Double> product2 = (Tensor.of("i0 x i1", tensor1, tensor2)*-100).softplus();
        then :
            product1.toString() == "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0); ->d[2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0)"
            product2.toString() == "[2x3]:(200.0, 200.0, 200.0, 200.0, 200.0, 200.0); ->d[2x3]:(-100.0, -100.0, -100.0, -100.0, -100.0, -100.0)"

        when : 'We perform a backwards pass of a gradient of `-0.1`:'
            product1.backward( -0.1 );
        then :
            tensor1.gradient.get().toString() == "[1x3]:(-20.0, -20.0, -20.0)"
            tensor2.gradient.get().toString() == "[2x1]:(60.0, 60.0)"

        when : 'We perform a backwards pass of a gradient of `-0.1`:'
            product2.backward( -0.1 );
        then :
            tensor1.gradient.get().toString() == "[1x3]:(-40.0, -40.0, -40.0)"
            tensor2.gradient.get().toString() == "[2x1]:(120.0, 120.0)"
    }

    def 'A function with expression "softplus(tanh(I[0]*I[1]*2)*I[1])" can be backpropagated.'()
    {
        given :
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true);
            Tensor<Double> tensor1 = Tensor.of(Shape.of(2), 2d);
            Tensor<Double> tensor2 = Tensor.of(Shape.of(2), 4d);
            tensor1.setRqsGradient(true);
            tensor2.setRqsGradient(true);

        when :
            Tensor<Double> product1 = Tensor.of("softplus(tanh(I[0]*I[1]*2)*I[1])", [tensor1, tensor2]);
            Tensor<Double> product2 = ((tensor1 * tensor2 * 2).tanh()*tensor2).softplus();
        then :
            product1.toString({it.hasDerivatives=false}) == "[2]:(4.01815, 4.01815)"
            product2.toString({it.hasDerivatives=false}) == "[2]:(4.01815, 4.01815)"

        when : 'We perform a backwards pass of a gradient of `100`:'
            product1.backward( 100 );
        then :
            tensor1.gradient.get().toString() == "[2]:(159.09e-12, 159.09e-12)"
            tensor2.gradient.get().toString() == "[2]:(98.2014, 98.2014)"

        when : 'We perform a backwards pass of a gradient of `100`:'
            product2.backward( 100 );
        then :
            tensor1.gradient.get().toString() == "[2]:(318.18e-12, 318.18e-12)"
            tensor2.gradient.get().toString() == "[2]:(196.403, 196.403)"
    }

}
