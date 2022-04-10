package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.assembly.ParseUtil
import neureka.ndim.config.types.views.SimpleReshapeView
import neureka.optimization.implementations.ADAM
import neureka.optimization.implementations.SGD
import neureka.view.TsrStringSettings
import spock.lang.Ignore
import spock.lang.Specification

//import org.junit.Test

class Optimizer_Spec extends Specification
{
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

    def 'The optimization function for the SGD algorithm produces the expected result'()
    {
        given : 'We use a common learning rate.'
            var learningRate = 0.01
        and : 'Based on that we instantiate the SGD optimization inline function.'
            var fun = Function.of("I[0] <- (-1 * (I[0] - $learningRate))")
        and : 'A tensor, which will be treated as gradient.'
            var g = Tsr.of(1.0)

        when : 'We apply the function to the gradient...'
            var result = fun(g)

        then : 'Both the result tensor and the gradient will have the expected value.'
            result.toString() == "(1):[-0.99]"
            g.toString() == "(1):[-0.99]"
        and : 'The result will be identical to the gradient, simply because its an inline function.'
            result === g
    }

    @Ignore
    def 'Conv dot based feed forward and activation produces expected result.'()
    {
        given :
            def data = Tsr.of([8, 4], [ // a-b*c
                                          1d,  2d,  2d, -3d,
                                          3d, -1d, -1d,  4d,
                                         -1d, -2d, -3d, -7d,
                                         -2d, -3d,  4d,  10d,
                                          4d,  5d, -1d,  9d,
                                          6d,  2d,  3d,  0d,
                                          7d,  3d,  2d,  1d,
                                         -4d, -4d,  2d,  4d
                                    ])

            var X = data[0..-1, 0..2].T()
        and :
            var w1 = Tsr.of([8, 3], ":-)").setRqsGradient(true)
            var w2 = Tsr.of([7, 8], "O.o").setRqsGradient(true)
            var w3 = Tsr.of([1, 7], ":P").setRqsGradient(true)

        expect:
            X.NDConf instanceof SimpleReshapeView
            X.NDConf.indicesMap()  == [ 8,1  ] as int[]
            X.NDConf.translation() == [ 1, 4 ] as int[]
            X.NDConf.offset()      == [ 0,0  ] as int[]
            X.NDConf.spread()      == [ 1,1  ] as int[]
            X.NDConf.shape()       == [ 3, 8 ] as int[]
        and:
            w1.toString() == "(8x3):[-0.97380, -0.07925, 1.78121, -1.39653, -2.01835, -0.41131, 0.64522, -2.5104, 1.40632, -0.51236, -0.62507, -0.09238, -0.52655, 0.57702, -1.37303, -1.27665, 0.63007, -2.15027, -1.12862, 1.48548, -0.37397, -0.51683, 1.14686, 0.87993]:g:[null]"
            w2.toString() == "(7x8):[-0.24678, 0.81013, 0.23709, -0.63630, 0.96606, 1.48405, 0.02693, 0.21423, -1.29268, 0.62331, -2.10922, 0.11949, -1.68853, 0.61275, 0.24238, 0.18039, -1.06208, 2.36855, -0.17225, -0.01213, 0.80261, -1.2896, -0.00226, -0.04938, 0.76502, 0.21277, -0.09264, 0.86399, -0.27399, -0.56845, -0.40595, -0.89906, -0.32320, 0.54387, -0.26504, -0.43978, 1.58936, 1.00546, 0.03889, 0.16153, -0.27125, -0.63222, 3.87467, 0.31497, -0.47855, -0.01106, -1.39322, 0.23283, 0.36686, 0.25312, ... + 6 more]:g:[null]"
            w3.toString() == "(1x7):[-0.68675, 1.09107, 0.20312, -1.18979, 0.53206, 2.11926, 1.15821]:g:[null]"
        and :
            w1.set(new ADAM<>(w1))
            w2.set(new SGD<>(0.01))

        when:
            var f = Function.of("softplus(I[0])")
        then :
            f.toString() == "softplus(I[0])"

        when :
            Tsr s = w1.convDot(X)
            Tsr a = f(s)
            Tsr b = f(w2.convDot(a))
            var y = w3.convDot(b)

        then:
            0.9 < ParseUtil.similarity(y.toString(), "(8):[0.31170, 26.1058, -0.76121, 82.0447, 0.22348, 26.5363, 6.02289, 57.1377]; ->d({Derivative=Derivative[null], Ends=Ends[[I@137855df]})")

        when :
            y.backward(-4)
            w1.applyGradient()
            w2.applyGradient()
            w3.applyGradient()
        then :
            w1.gradient.toString() == "(8x3):[-0.97380, -0.07925, 1.78121, -1.39653, -2.01835, -0.41131, 0.64522, -2.5104, 1.40632, -0.51236, -0.62507, -0.09238, -0.52655, 0.57702, -1.37303, -1.27665, 0.63007, -2.15027, -1.12862, 1.48548, -0.37397, -0.51683, 1.14686, 0.87993]:g:[-22.2834, -28.5797, 31.7585, -25.9093, -31.9633, 37.8158, -176.346, 175.956, -321.819, -50.9334, -68.3452, 54.9529, -1.40994, -10.1392, -11.3331, 3.39221, 3.06355, -2.62521, -0.99570, -0.63634, 8.29206, -3.4819, -3.20387, -0.94523]"
            w2.gradient.toString() == "(7x8):[-0.24678, 0.81013, 0.23709, -0.63630, 0.96606, 1.48405, 0.02693, 0.21423, -1.29268, 0.62331, -2.10922, 0.11949, -1.68853, 0.61275, 0.24238, 0.18039, -1.06208, 2.36855, -0.17225, -0.01213, 0.80261, -1.2896, -0.00226, -0.04938, 0.76502, 0.21277, -0.09264, 0.86399, -0.27399, -0.56845, -0.40595, -0.89906, -0.32320, 0.54387, -0.26504, -0.43978, 1.58936, 1.00546, 0.03889, 0.16153, -0.27125, -0.63222, 3.87467, 0.31497, -0.47855, -0.01106, -1.39322, 0.23283, 0.36686, 0.25312, ... + 6 more]:g:[null]"
            w3.gradient.toString() == "(1x7):[-0.68675, 1.09107, 0.20312, -1.18979, 0.53206, 2.11926, 1.15821]:g:[null]"

        when :
            y.backward(-2)
            w1.applyGradient()
            w2.applyGradient()
            w3.applyGradient()
        then :
            w1.gradient.toString() == "(8x3):[-0.97380, -0.07925, 1.78121, -1.39653, -2.01835, -0.41131, 0.64522, -2.5104, 1.40632, -0.51236, -0.62507, -0.09238, -0.52655, 0.57702, -1.37303, -1.27665, 0.63007, -2.15027, -1.12862, 1.48548, -0.37397, -0.51683, 1.14686, 0.87993]:g:[null]"
            w2.gradient.toString() == "(7x8):[-0.24678, 0.81013, 0.23709, -0.63630, 0.96606, 1.48405, 0.02693, 0.21423, -1.29268, 0.62331, -2.10922, 0.11949, -1.68853, 0.61275, 0.24238, 0.18039, -1.06208, 2.36855, -0.17225, -0.01213, 0.80261, -1.2896, -0.00226, -0.04938, 0.76502, 0.21277, -0.09264, 0.86399, -0.27399, -0.56845, -0.40595, -0.89906, -0.32320, 0.54387, -0.26504, -0.43978, 1.58936, 1.00546, 0.03889, 0.16153, -0.27125, -0.63222, 3.87467, 0.31497, -0.47855, -0.01106, -1.39322, 0.23283, 0.36686, 0.25312, ... + 6 more]:g:[null]"
            w3.gradient.toString() == "(1x7):[-0.68675, 1.09107, 0.20312, -1.18979, 0.53206, 2.11926, 1.15821]:g:[null]"

    }

}
