package unit.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.optimization.Optimizer
import neureka.optimization.implementations.ADAM
import spock.lang.Shared
import spock.lang.Specification

class ADAM_Tests extends Specification
{
    @Shared Tsr w = new Tsr(0)
    @Shared Optimizer o = new ADAM(w)

    def setup() {
        w.add(o)
    }


    def 'ADAM optimizes according to expected inputs' (
            int input, double gradient
    ) {
        given :
            Neureka.instance().reset()
            Tsr g = new Tsr(gradient)
        and :
            w.add( new Tsr( input ) )
            w.applyGradient()
        expect :
            w.toString().contains(g.toString())
            w.shape().hashCode()==g.shape().hashCode()
            w.translation().hashCode()==g.translation().hashCode()
            w.idxmap().hashCode()==g.idxmap().hashCode()
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
    }


    def 'Equations used by ADAM return expected result.' (
            String equation, List<Object> inputs, String expected
    ) {
        given :
            Neureka.instance().reset()
            def t = new Tsr(equation, inputs)
        expect :
            t.toString().contains(expected)

        where :
            equation | inputs | expected
            "( 1 - I[0]) * I[1]"                       | [0.9, -3]                           | "(1):[-0.29999E0]"
            "I[0] * I[1] + (1 - I[2]) * I[3]"          | [0.9, 0, 0.9, -3]                   | "(1):[-0.29999E0]" //-> I[0] below
            "I[0] / ( 1 - I[1] )"                      | [-0.3, 0.9]                         | "(1):[-3.0]" // -> I[2] below below!
            "I[0] ^ 0.5 + I[1]"                        | [9, 1e-7]                           | "(1):[3.0]"
            "I[0] - I[1] * I[2] /( I[3] ^ 0.5 + I[4] )"| [0, 0.01, -3, 9, 1e-7 ]             | "(1):[0.00999E0]"
    }


    def 'Equations "I[0]*I[1]+(1-I[2])*I[3]" and "(1-I[0])*I[1]" used within ADAM return expected results.' (
            String equation, double[] input, double output
    ) {
        given : Function f = Function.create(equation)
        expect : output == f(input)
        where :
        equation                  | input                             || output
        "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{0.9, 0.0, 0.9, -3.0} || -0.29999999999999993
        "I[0]*I[1]+(1-I[2])*I[3]" | new double[]{-0.9, 2.0, 0.2, 1.0} || -1.0
        "(1-I[0])*I[1]"           | new double[]{0.9, -3.0}           || -0.29999999999999993
        "(1-I[0])*I[1]"           | new double[]{-0.9, 2.0}           || 3.8
    }


}
