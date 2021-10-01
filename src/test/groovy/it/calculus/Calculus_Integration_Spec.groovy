package it.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import spock.lang.Specification

class Calculus_Integration_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> Calculus Integration Tests </h2>
            <p>
                Specified below are strict tests covering the behavior
                of the classes located within the calculus package.
            </p>
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Tensor results of various Function instances return expected results.'(
            String equation, List<Tsr> inputs, Integer index, Map<List<Integer>,List<Double>> expected
    ) {
        given : "A new Function instance created from ${equation}."
            Function f = new FunctionBuilder( Neureka.get().context() ).build(equation, true) // TODO : test with 'doAD' : false!

        and : 'The result is being calculated by invoking the Function instance.'
            Tsr<?> result = ( index != null )
                ? f.derive( inputs, index )
                : f.call(   inputs        )
            List<Double> value = ( index != null )
                                    ? (result.value64() as List<Double>)
                                    : (result.value64() as List<Double>)

        expect : "The calculated result ${result} should be equal to expected ${expected}."
            value == expected.values().first()

        and : 'The shape is as expected as well : '
            result.shape() == expected.keySet().first()

        // Todo: unrecognized operation throws exception that is not recursion error
        where :
            equation                         | inputs                                                         | index || expected
            "quad(sumJs(Ij))"                | [Tsr.of([2],[1.0, 2.0]), Tsr.of([2],[3.0, -5.0])]              | null  || [[2]:[16.0, 9.0]]
            "tanh(sumJs(Ij))"                | [Tsr.of([2],[1.0, 2.0]), Tsr.of([2],[3.0, -4.0])]              | null  || [[2]:[0.9701425001453319, -0.8944271909999159]]
            "softplus(prodJs(Ij-2))"         | [Tsr.of([2],[1.0, 2.0]), Tsr.of([2],[3.0, -4.0])]              | null  || [[2]:[0.31326168751822286, 0.6931471805599453]]
            "softplus([-1, 0, -2, -2](Ij-2))"| [Tsr.of([2, 4], [10, 12, 16, 21, 33, 66, 222, 15])]            | null  || [[1,2,2,2]:[8.000335406372896, 10.000045398899216, 14.000000831528373, 19.000000005602796, 31.000000000000032, 64.0, 220.0, 13.000002260326852]]
            "softplus(i0*i1)*i2"             | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[-0.0018221023888012912, 0.2845552390654007]]
            "sumJs(ij^3)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[(-1+7*7*7+2*2*2), (3*3*3+-1+2*2*2)]]
            "sumJs(ij^3)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[3*Math.pow(7, 2), 3*Math.pow(-1, 2)]]
            "sumJs(ij*ij)"                   | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[(1+7*7+2*2), (3*3+1+2*2)]]
            "sumJs(ij*ij)"                   | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[2*Math.pow(7, 1), 2*Math.pow(-1, 1)]]
            "sumJs(ij/2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[4,2]]
            "sumJs(ij/2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[0.5, 0.5]]
            "sumJs(ij+2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[(1+9+4), (5+1+4)]]
            "sumJs(ij+2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[1.0, 1.0]]
            "sumJs(ij-2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[(-3+5+0), (1+-3+0)]]
            "sumJs(ij-2)"                    | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[1.0, 1.0]]
            "sumJs(sumJs(ij))"               | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[8.0*3.0, 4.0*3.0]]
            "sumJs(sumJs(ij))"               | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[3.0, 3.0]]
            "sumJs(prodJs(ij))"              | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[(-14)*3, (-6)*3]]
            "(prodJs(ij))"                   | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[-2.0, 6.0]]
            "-(prodJs(ij))%3"                | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | null  || [[2]:[-(-14)%3, -0.0]]
            "sumJs(prodJs(ij))"              | [Tsr.of([2],[-1, 3]), Tsr.of([2],[7, -1]), Tsr.of([2],[2, 2])] | 1     || [[2]:[-2.0*3, 6.0*3]]
            "relu(I[0])"                     | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         | null  || [[2,3]:[-0.04, 7.0, -0.01, 2.0, 3.0, 8.0]]
            "relu(I[0])"                     | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         |  0    || [[2,3]:[0.01, 1.0, 0.01, 1.0, 1.0, 1.0]]
            "quad(I[0])"                     | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         | null  || [[2,3]:[16.0, 49.0, 1.0, 4.0, 9.0, 64.0]]
            "quad(I[0])"                     | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         |  0    || [[2,3]:[-8.0, 14.0, -2.0, 4.0, 6.0, 16.0]]
            "abs(I[0])"                      | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         | null  || [[2,3]:[4.0, 7.0, 1.0, 2.0, 3.0, 8.0]]
            "abs(I[0])"                      | [Tsr.of([2, 3], [-4, 7, -1, 2, 3, 8])]                         |  0    || [[2,3]:[-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]]
            "dimtrim(I[0])"                  | [Tsr.of([1, 3, 1], [1, 2, 3])]                                 | null  || [[3]:[1, 2, 3]]
            "dimtrim(I[0])"                  | [Tsr.of([1, 3, 1, 1], [-4, 2, 5])]                             | null  || [[3]:[-4, 2, 5]]
            "ln(i0)"                         | [Tsr.of(3)]                                                    | null  || [[1]:[Math.log(3)]]
            "ln(i0)"                         | [Tsr.of(3)]                                                    |  0    || [[1]:[0.3333333333333333]]
    }


    def 'Reshaping on 3D tensors works by instantiate a Function instance built from a String.'()
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Function f = Function.of("[2, 0, 1]:(I[0])")

        when : Tsr t = Tsr.of([3, 4, 2], 1..5)
        then : t.toString().contains("[3x4x2]:(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)")

        when : Tsr r = f(t)
        then :
            r.toString().contains("[2x3x4]")
            r.toString().contains("[2x3x4]:(1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0)")
    }


    def 'The "DimTrim" operation works forward as well as backward!'(){

        given :
            Tsr t = Tsr.of([1, 1, 3, 2, 1], 8).setRqsGradient(true)

        when :
            Tsr trimmed = Function.of("dimtrim(I[0])")(t)

        then :
            trimmed.toString().contains("(3x2):[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]; ->d({")

        when :
            Tsr back = trimmed.backward()

        then :
            back == trimmed
        and :
            t.getGradient().toString() == "(1x1x3x2x1):[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
    }



}
