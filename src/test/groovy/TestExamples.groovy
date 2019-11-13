import neureka.core.Tsr;
import neureka.core.function.Function;
import neureka.core.function.factory.assembly.FunctionBuilder;
import neureka.core.function.factory.autograd.GraphNode;
import org.junit.Test;
import util.NTester_Function;
import util.NTester_Tensor;

class TestExamples {

    @Test
    void testReadmeExamples()
    {
        NTester_Tensor tester = new NTester_Tensor("Tensor tester (only cpu)")

        Tsr x = new Tsr([1], 3).setRqsGradient(true)
        Tsr b = new Tsr([1], -4)
        Tsr w = new Tsr([1], 2)
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        Tsr y = new Tsr([x, b, w], "((i0+i1)*i2)^2")
        tester.testTensor(y, "[1]:(4.0); ->d[1]:(-8.0), ")
        y.backward(new Tsr(2))
        y = b + w * x
        System.out.println(y)
        //tester.testTensor(x, ["-16.0"])

    }

}
