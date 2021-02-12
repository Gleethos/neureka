package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.optimization.Optimizer
import neureka.optimization.implementations.ADAM
import neureka.utility.TsrAsString
import spock.lang.Shared
import spock.lang.Specification

class ADAM_Tests extends Specification
{
    @Shared Tsr w = new Tsr(0)
    @Shared Optimizer o = new ADAM(w)

    def setupSpec()
    {
        reportHeader """
                <h2> ADAM Optimizer Behavior </h2>
                <br> 
                <p>
                    This specification check the behavior of the ADAM class.        
                </p>
            """
    }

    def setup() {
        Neureka.instance().reset()
        w.set(o)
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = "dgc"
    }


    def 'ADAM optimizes according to expected inputs' (
            int input, double gradient
    ) {
        given : 'A new scalar gradient tensor is being created.'
            Tsr g = new Tsr(gradient)
        and : 'The following input is being applied to the tensor (and internal optimizer)...'
            w.set( new Tsr( input ) )
            w.applyGradient()

        expect : 'The following state emerges:'
            w.toString().contains(g.toString())
            w.shape().hashCode()==g.shape().hashCode()
            w.translation().hashCode()==g.translation().hashCode()
            w.indicesMap().hashCode()==g.indicesMap().hashCode()
            w.spread().hashCode()==g.spread().hashCode()
            w.offset().hashCode()==g.offset().hashCode()

        where :
            input | gradient
             -3   | 0.009999999666666677
             -3   | 0.02343838820965563
              2   | 0.030115667802083777
             -3   | 0.040571568761377755
              2   | 0.04604647775568702
              2   | 0.04750863243746767
             -4   | 0.054017821302644514
             -3   | 0.06320597194500503
             -3   | 0.07446842921800374
              2   | 0.08205723598079066
              //6   | 0.08203818944011058
              //6   | 0.0770990337952347
              //6   | 0.06869002023261302
              //16  | 0.056165924397208
              //16  | 0.04113435163034236
              //200 | 0.02945128411568828
              //220 | 0.014971197286174196
              //250 | -0.0014452178809236885
              //255 | -0.01929025587848216
    }


    def 'Equations used by ADAM return expected result.' (
            String expression, List<Object> inputs, String expected
    ) {
        when : 'A new tensor is being created from the given equation and array of input tensors...'
            def t = new Tsr( expression, inputs )
        then : '...this produces the expected result String.'
            t.toString().contains( expected )

        where : 'The following expressions, inputs and expected String results are being used :'
            expression                                 | inputs                  || expected
            "( 1 - I[0]) * I[1]"                       | [0.9, -3]               || "(1):[-0.29999E0]"
            "I[0] * I[1] + (1 - I[2]) * I[3]"          | [0.9, 0, 0.9, -3]       || "(1):[-0.29999E0]" //-> I[0] below
            "I[0] / ( 1 - I[1] )"                      | [-0.3, 0.9]             || "(1):[-3.0]" //-> I[2] below below!
            "I[0] ^ 0.5 + I[1]"                        | [9, 1e-7]               || "(1):[3.0]"
            "I[0] - I[1] * I[2] /( I[3] ^ 0.5 + I[4] )"| [0, 0.01, -3, 9, 1e-7 ] || "(1):[0.00999E0]"
    }


    def 'Equations "I[0]*I[1]+(1-I[2])*I[3]" and "(1-I[0])*I[1]" used within ADAM return expected results.' (
            String expression, double[] input, double output
    ) {
        given : Function f = Function.create(expression)
        expect : output == f(input)
        where : 'The following expressions, inputs and expected String results are being used :'
            expression                | input                             || output
            "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{0.9, 0.0, 0.9, -3.0} || -0.29999999999999993
            "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{-0.9, 2.0, 0.2, 1.0} || -1.0
            "(1-I[0])*I[1]"           | new double[]{0.9, -3.0}           || -0.29999999999999993
            "(1-I[0])*I[1]"           | new double[]{-0.9, 2.0}           || 3.8
    }


}
