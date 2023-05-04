package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.math.Function
import neureka.optimization.Optimizer
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Subject

@Narrative('''

    ADAM is a more powerful alternative to the classical stochastic gradient descent. 
    It combines the best properties of the AdaGrad and the RMSProp algorithms, which makes 
    it especially well suited for sparse gradients and noisy data.
    Adam is the most popular among the adaptive optimizers
    because its adaptive learning rate working so well with sparse datasets.
   
''')
@Subject([Optimizer])
class ADAM_Spec extends Specification
{
    @Shared Tsr<?> w = Tsr.of(0d)
    @Shared Optimizer<?> o = Optimizer.ADAM.create(w)

    def setupSpec()
    {
        reportHeader """
                The code below assumes that for we
                have the following 2 variables setup
                throughout every data table iteration:
                ```
                    Tsr<?> w = Tsr.of(0d)
                    Optimizer<?> o = Optimizer.ADAM.create(w)                   
                    w.set(o)                 
                ```
            """
    }

    def setup() {
        Neureka.get().reset()
        w.set(o)
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


    def 'ADAM optimizes according to expected inputs' (
            int gradient, double expectedWeight
    ) {
        given : 'A new scalar gradient tensor is being created.'
            Tsr g = Tsr.of(expectedWeight)
        and : 'The following input is being applied to the tensor (and internal optimizer)...'
            w.set( Tsr.of( (double)gradient ) )
            w.applyGradient()

        expect : 'The following state emerges:'
            w.toString().contains(g.toString())
            w.shape.hashCode()==g.shape.hashCode()
            w.strides().hashCode()==g.strides().hashCode()
            w.indicesMap().hashCode()==g.indicesMap().hashCode()
            w.spread().hashCode()==g.spread().hashCode()
            w.offset().hashCode()==g.offset().hashCode()

        where :
            gradient | expectedWeight
             -3      | 0.009999999666666677
             -3      | 0.01999
              2      | 0.02426
             -3      | 0.03034
              2      | 0.03332
              2      | 0.03409
             -4      | 0.03738
             -3      | 0.04194
             -3      | 0.04744
              2      | 0.05112
    }


    def 'Equations used by ADAM return expected result.' (
            String expression, Double[] inputs, String expected
    ) {
        given : 'We create tensors given an equation and array or list of input tensors...'
            var t1 = Tsr.of( expression, inputs )
            var t2 = Tsr.of( expression, inputs as Float[] )
            var t3 = Tsr.of( expression, true, inputs.collect( it -> Tsr.of(it) ) )
            var t4 = Tsr.of( expression, false, inputs.collect( it -> Tsr.of(it) ) )
            var t5 = Tsr.of( expression, false, inputs.collect( it -> Tsr.of(it) ) as Tsr<Double>[] )

        expect : '...this produces the expected result String.'
            t1.toString().contains( expected )
            t2.toString().contains( expected.replace(".29999", ".30000") )
            t3.toString().contains( expected )
            t4.toString().contains( expected )
            t5.toString().contains( expected )

        where : 'The following expressions, inputs and expected String results are being used :'
            expression                                  | inputs                       || expected
            "( 1 - I[0]) * I[1]"                        | [0.9d, -3d]                  || "(1):[-0.29999]"
            "I[0] * I[1] + (1 - I[2]) * I[3]"           | [0.9d, 0d, 0.9d, -3d]        || "(1):[-0.29999]" //-> I[0] below
            "I[0] / ( 1 - I[1] )"                       | [-0.3d, 0.9d]                || "(1):[-3.0]" //-> I[2] below below!
            "I[0] ** 0.5 + I[1]"                        | [9d, 1e-7d]                  || "(1):[3.0]"
            "I[0] - I[1] * I[2] /( I[3] ** 0.5 + I[4] )"| [0d, 0.01d, -3d, 9d, 1e-7d ] || "(1):[0.00999]"
    }


    def 'Equations "I[0]*I[1]+(1-I[2])*I[3]" and "(1-I[0])*I[1]" used within ADAM return expected results.' (
            String expression, double[] input, double output
    ) {
        given : Function f = Function.of(expression)
        expect : output == f(input)
        where : 'The following expressions, inputs and expected String results are being used :'
            expression                | input                             || output
            "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{0.9, 0.0, 0.9, -3.0} || -0.29999999999999993
            "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{-0.9, 2.0, 0.2, 1.0} || -1.0
            "(1-I[0])*I[1]"           | new double[]{0.9, -3.0}           || -0.29999999999999993
            "(1-I[0])*I[1]"           | new double[]{-0.9, 2.0}           || 3.8
    }


}
