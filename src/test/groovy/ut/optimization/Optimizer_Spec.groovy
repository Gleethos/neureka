package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.ndim.config.types.views.SimpleReshapeView
import neureka.optimization.Optimizer
import neureka.optimization.implementations.ADAM
import neureka.optimization.implementations.SGD
import neureka.view.NDPrintSettings
import spock.lang.Ignore
import spock.lang.Specification
import spock.lang.Subject

@Subject([Optimizer])
class Optimizer_Spec extends Specification
{
    def setup() {
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

    @Ignore
    def 'Dot based feed forward and activation produces expected result.'()
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
            w1.items.collect({it.round(2)}) == [-0.22, 0.12, -0.65, -0.16, 0.96, -1.5, -0.33, 0.16, -0.62, -0.18, -1.24, 0.48, -0.41, 0.19, -0.58, -0.19, 0.41, 0.7, -0.47, 0.2, -0.54, -0.2, -0.65, -1.23]
            w2.items.collect({it.round(2)}) == [-0.5, -0.36, 0.7, 0.35, 0.83, -0.32, 1.81, 0.36, 0.75, 0.34, 0.78, -0.33, -0.47, 0.77, 0.79, 0.33, 0.73, -0.34, -0.23, 0.59, 0.83, 0.31, 0.68, -0.35, -2.29, -1.63, 0.87, 0.28, 0.62, -0.34, -0.84, 0.5, 0.91, 0.25, 0.55, -0.33, 0.16, -0.7, 0.94, 0.22, 0.49, -0.31, -1.14, -0.43, 0.96, 0.18, 0.41, -0.28, 1.1, 0.79, 0.98, 0.14, 0.32, -0.24, -0.72, 0.91]
            w3.items.collect({it.round(2)}) == [0.43, 0.38, 1.21, -0.12, 0.83, 1.86, 0.5]
        and :
            w1.set(new ADAM<>(w1))
            w2.set(new SGD<>(0.01))

        when:
            var f = Function.of("softplus(I[0])")
        then :
            f.toString() == "softplus(I[0])"

        when :
            Tsr s = w1.dot(X)
            Tsr a = f(s)
            Tsr b = f(w2.dot(a))
            var y = w3.dot(b)

        then:
            y.items.collect({it.round(2)}) == [5.33, 6.14, 18.8, 7.01, 8.02, 4.22, 4.07, 9.36]

        when :
            y.backward(-4)
            w1.applyGradient()
            w2.applyGradient()
            w3.applyGradient()
        then :
            w1.gradient.items.collect({it.round(2)}) == []
            w2.gradient.items.collect({it.round(2)})== []
            w3.gradient.items.collect({it.round(2)}) == []

        when :
            y.backward(-2)
            w1.applyGradient()
            w2.applyGradient()
            w3.applyGradient()
        then :
            w1.gradient.items.collect({it.round(2)}) == []
            w2.gradient.items.collect({it.round(2)}) == []
            w3.gradient.items.collect({it.round(2)}) == []

    }

}
