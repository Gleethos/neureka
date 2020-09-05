package it.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.frontend.assembly.FunctionBuilder
import spock.lang.Specification

class Calculus_Integration_Tests extends Specification
{

    def 'Tensor results of various Function instances return expected results.'(
            String equation, List<Tsr> inputs, Integer index, List<Double> expected
    ) {
        given : "A new Function instance created from ${equation}."
        Function f = FunctionBuilder.build(equation, true) // TODO : test with 'doAD' : false!

        and : 'The result is being calculated by invoking the Function instance.'
        def result = (index!=null)
                ? (f.derive( inputs, index ).value64() as List<Double>)
                : (f.call(   inputs        ).value64() as List<Double>)

        expect : "The calculated result ${result} should be equal to expected ${expected}."
        result==expected

        // Todo: unrecognized operation throws exception that is not recursion error
        where :
        equation                         | inputs                                                            | index || expected
        "quad(sumJs(Ij))"                | [new Tsr([2],[1.0, 2.0]), new Tsr([2],[3.0, -5.0])]               | null  || [16.0, 9.0]
        "tanh(sumJs(Ij))"                | [new Tsr([2],[1.0, 2.0]), new Tsr([2],[3.0, -4.0])]               | null  || [0.9701425001453319, -0.8944271909999159]
        "softplus(prodJs(Ij-2))"         | [new Tsr([2],[1.0, 2.0]), new Tsr([2],[3.0, -4.0])]               | null  || [0.31326168751822286, 0.6931471805599453]
        "softplus([-1, 0, -2, -2](Ij-2))"| [new Tsr([2, 4], [10, 12, 16, 21, 33, 66, 222, 15])]              | null  || [8.000335406372896, 10.000045398899216, 14.000000831528373, 19.000000005602796, 31.000000000000032, 64.0, 220.0, 13.000002260326852]
        "softplus(i0*i1)*i2"             | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [-0.0018221023888012912, 0.2845552390654007]
        "sumJs(ij^3)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [(-1+7*7*7+2*2*2), (3*3*3+-1+2*2*2)]
        "sumJs(ij^3)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [3*Math.pow(7, 2), 3*Math.pow(-1, 2)]
        "sumJs(ij*ij)"                   | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [(1+7*7+2*2), (3*3+1+2*2)]
        "sumJs(ij*ij)"                   | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [2*Math.pow(7, 1), 2*Math.pow(-1, 1)]
        "sumJs(ij/2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [4,2]
        "sumJs(ij/2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [0.5, 0.5]
        "sumJs(ij+2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [(1+9+4), (5+1+4)]
        "sumJs(ij+2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [1.0, 1.0]
        "sumJs(ij-2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [(-3+5+0), (1+-3+0)]
        "sumJs(ij-2)"                    | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [1.0, 1.0]
        "sumJs(sumJs(ij))"               | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [8.0*3.0, 4.0*3.0]
        "sumJs(sumJs(ij))"               | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [3.0, 3.0]
        "sumJs(prodJs(ij))"              | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [(-14)*3, (-6)*3]
        "(prodJs(ij))"                   | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [-2.0, 6.0]
        "-(prodJs(ij))%3"                | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | null  || [-(-14)%3, -0.0]
        "sumJs(prodJs(ij))"              | [new Tsr([2],[-1, 3]), new Tsr([2],[7, -1]), new Tsr([2],[2, 2])] | 1     || [-2.0*3, 6.0*3]
        "relu(I[0])"                     | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           | null  || [-0.04, 7.0, -0.01, 2.0, 3.0, 8.0]
        "relu(I[0])"                     | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           |  0    || [0.01, 1.0, 0.01, 1.0, 1.0, 1.0]
        "quad(I[0])"                     | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           | null  || [16.0, 49.0, 1.0, 4.0, 9.0, 64.0]
        "quad(I[0])"                     | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           |  0    || [-8.0, 14.0, -2.0, 4.0, 6.0, 16.0]
        "abs(I[0])"                      | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           | null  || [4.0, 7.0, 1.0, 2.0, 3.0, 8.0]
        "abs(I[0])"                      | [new Tsr([2, 3], [-4, 7, -1, 2, 3, 8])]                           |  0    || [-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]

    }


    def 'Reshaping on 3D tensors works by instantiate a Function instance built from a String.'()
    {
        given :
            Neureka.instance().reset()
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
            Function f = Function.create("[2, 0, 1]:(I[0])")

        when : Tsr t = new Tsr([3, 4, 2], 1..5)
        then : t.toString().contains("[3x4x2]:(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)")

        when : Tsr r = f(t)
        then :
            r.toString().contains("[2x3x4]")
            r.toString().contains("[2x3x4]:(1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0)")
    }

}
