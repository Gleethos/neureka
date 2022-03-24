package it.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import neureka.devices.Device
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
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
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    @IgnoreIf({ device == 'GPU' && !Neureka.get().canAccessOpenCL() }) // We need to assure that this system supports OpenCL!
    def 'Tensor results of various Function instances return expected results.'(
            String equation, List<Tsr> inputs, Integer index, Map<List<Integer>,List<Double>> expected
    ) {
        given : "A new Function instance created from ${equation}."
            Function f = new FunctionBuilder( Neureka.get().backend() ).build(equation, true) // TODO : test with 'doAD' : false!
        and :
            inputs.each {it.to(Device.find(device))}

        and : 'The result is being calculated by invoking the Function instance.'
            Tsr<?> result = ( index != null ? f.derive( inputs, index ) : f.call( inputs ) )
            List<Double> value = result.getValueAs(double[].class) as List<Double>

        expect : "The calculated result ${result} should be equal to expected ${expected}."
            value == expected.values().first()

        and : 'The shape is as expected as well : '
            result.shape == expected.keySet().first()

        // Todo: unrecognized operation throws exception that is not recursion error
        where :
            device | equation                         | inputs                                                           | index || expected
            'CPU'  | "quad(sumJs(Ij))"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -5d])]                    | null  || [[2]:[16d, 9d]]
            'GPU'  | "quad(sumJs(Ij))"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -5d])]                    | null  || [[2]:[16d, 9d]]
            'CPU'  | "tanh(sumJs(Ij))"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.9993292997390673, -0.9640275800758169]]
            'GPU'  | "tanh(sumJs(Ij))"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.9993293285369873, -0.9640275835990906]]
            'CPU'  | "tanh(i0*i1)"                    | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | 0     || [[2]:[0.0295981114963193, -1.8005623907413337E-6]]
            'GPU'  | "tanh(i0*i1)"                    | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | 0     || [[2]:[0.0295981764793396, -1.9073486328125E-6]]
            'CPU'  | "fast_tanh(i0*i1)"               | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.9486832980505138, -0.9922778767136677]]
            'CPU'  | "fast_tanh(i0*i1)"               | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | 0     || [[2]:[0.09486832980505137, -0.00763290674395129]]
            'CPU'  | "softsign(i0*i1)"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.75, -0.8888888888888888]]
            'CPU'  | "softsign(i0*i1)"                | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | 0     || [[2]:[0.1875, -0.04938271604938271]]
            'CPU'  | "fast_gaus(i0/i1)"               | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.8999999999999999, 0.8]]
            'CPU'  | "fast_gaus(i0/i1)"               | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | 0     || [[2]:[-0.18, -0.16]]
            'CPU'  | "softplus(prodJs(Ij-2))"         | [Tsr.of([2],[1d, 2d]), Tsr.of([2],[3d, -4d])]                    | null  || [[2]:[0.31326168751822286, 0.6931471805599453]]
            'CPU'  | "softplus([-1, 0, -2, -2](Ij-2))"| [Tsr.of([2, 4], [10d,12d,16d,21d,33d,66d,222d,15d])]             | null  || [[1,2,2,2]:[8.000335406372896, 10.000045398899218, 14.000000831528373, 19.000000005602796, 31.000000000000036, 64.0, 220.0, 13.000002260326852]]
            'CPU'  | "softplus(i0*i1)*i2"             | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[-0.0018221023888012908, 0.2845552390654007]]
            'CPU'  | "sumJs(ij^3)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[(-1+7*7*7+2*2*2), (3*3*3+-1+2*2*2)]]
            'CPU'  | "sumJs(ij^3)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[3*Math.pow(7, 2), 3*Math.pow(-1, 2)]]
            'CPU'  | "sumJs(ij*ij)"                   | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[(1+7*7+2*2), (3*3+1+2*2)]]
            'CPU'  | "sumJs(ij*ij)"                   | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[2*Math.pow(7, 1), 2*Math.pow(-1, 1)]]
            'CPU'  | "sumJs(ij/2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[4,2]]
            'CPU'  | "sumJs(ij/2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[0.5, 0.5]]
            'CPU'  | "sumJs(ij+2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[(1+9+4), (5+1+4)]]
            'CPU'  | "sumJs(ij+2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[1.0, 1.0]]
            'CPU'  | "sumJs(ij-2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[(-3+5+0), (1+-3+0)]]
            'CPU'  | "sumJs(ij-2)"                    | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[1.0, 1.0]]
            'CPU'  | "sumJs(sumJs(ij))"               | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[8.0*3.0, 4.0*3.0]]
            'CPU'  | "sumJs(sumJs(ij))"               | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[3.0, 3.0]]
            'CPU'  | "sumJs(prodJs(ij))"              | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[(-14)*3, (-6)*3]]
            'CPU'  | "(prodJs(ij))"                   | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[-2.0, 6.0]]
            'CPU'  | "-(prodJs(ij))%3"                | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| null  || [[2]:[-(-14)%3, -0.0]]
            'CPU'  | "sumJs(prodJs(ij))"              | [Tsr.of([2],[-1d,3d]),Tsr.of([2],[7d, -1d]), Tsr.of([2],[2d,2d])]| 1     || [[2]:[-2.0*3, 6.0*3]]
            'CPU'  | "relu(I[0])"                     | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     | null  || [[2,3]:[-0.04, 7.0, -0.01, 2.0, 3.0, 8.0]]
            'CPU'  | "relu(I[0])"                     | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     |  0    || [[2,3]:[0.01, 1.0, 0.01, 1.0, 1.0, 1.0]]
            'CPU'  | "quad(I[0])"                     | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     | null  || [[2,3]:[16.0, 49.0, 1.0, 4.0, 9.0, 64.0]]
            'CPU'  | "quad(I[0])"                     | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     |  0    || [[2,3]:[-8.0, 14.0, -2.0, 4.0, 6.0, 16.0]]
            'CPU'  | "abs(I[0])"                      | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     | null  || [[2,3]:[4.0, 7.0, 1.0, 2.0, 3.0, 8.0]]
            'CPU'  | "abs(I[0])"                      | [Tsr.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                     |  0    || [[2,3]:[-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]]
            'CPU'  | "dimtrim(I[0])"                  | [Tsr.of([1, 3, 1],    [ 1d, 2d, 3d])]                            | null  || [[3]:[1, 2, 3]]
            'CPU'  | "dimtrim(I[0])"                  | [Tsr.of([1, 3, 1, 1], [-4d, 2d, 5d])]                            | null  || [[3]:[-4, 2, 5]]
            'CPU'  | "ln(i0)"                         | [Tsr.of(3d)]                                                     | null  || [[1]:[Math.log(3)]]
            'CPU'  | "ln(i0)"                         | [Tsr.of(3d)]                                                     |  0    || [[1]:[0.3333333333333333]]
            'CPU'  | "selu(I[0])"                     | [Tsr.ofDoubles().withShape(3).all(-1)]                           | null  || [[3]:[-1.1113307378125625, -1.1113307378125625, -1.1113307378125625]]
            'GPU'  | "selu(I[0])"                     | [Tsr.ofDoubles().withShape(3).all(-1)]                           | null  || [[3]:[-1.1113307476043701, -1.1113307476043701, -1.1113307476043701]]
    }


    def 'Reshaping on 3D tensors works by instantiate a Function instance built from a String.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Function f = Function.of("[2, 0, 1]:(I[0])")

        when : Tsr t = Tsr.of([3, 4, 2], 1d..5d)
        then : t.toString().contains("[3x4x2]:(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)")

        when : Tsr r = f(t)
        then :
            r.toString().contains("[2x3x4]")
            r.toString().contains("[2x3x4]:(1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0)")
    }


    def 'The "DimTrim" operation works forward as well as backward!'()
    {
        given :
            Tsr t = Tsr.of([1, 1, 3, 2, 1], 8d).setRqsGradient(true)

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

    def 'Executed tensors are intermediate tensors.'() {

        given :
            var fun = Function.of('i0 * relu(i0) + 1')
        and :
            var t = Tsr.of(1f, -5f, -3f, 2f, 8f)
        expect :
            t.valueClass == Float
            fun.toString() == "((I[0] * relu(I[0])) + 1.0)"

        when :
            var result1 = fun.call(t)
            var result2 = fun.invoke(t)
            var result3 = fun.execute(t)

        then :
            !result1.isIntermediate()
        and :
            !result2.isIntermediate()
        and :
            result3.isIntermediate()
        and :
            result1.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"
            result2.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"
            result3.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"



    }



}
