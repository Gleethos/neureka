package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.optimization.Optimizer
import neureka.optimization.implementations.Momentum
import neureka.view.NDPrintSettings
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Subject

@Subject([Optimizer])
class Momentum_Spec extends Specification
{
    @Shared Tsr<Double> w = Tsr.of(0)
    @Shared Optimizer<Double> o = new Momentum<>(w)

    def setupSpec()
    {
        reportHeader """
                <h2> Momentum Optimizer Behavior </h2>
                <br> 
                <p>
                    This specification check the behavior of the Momentum class.        
                </p>
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


    def 'Momentum optimizes according to expected inputs' (
            int gradient, double expectedWeight
    ) {
        given : 'A new scalar gradient tensor is being created.'
            Tsr g = Tsr.of(expectedWeight)
        and : 'The following input is being applied to the tensor (and internal optimizer)...'
            w.set( Tsr.of( gradient ) )
            w.applyGradient()

        expect : 'The following state emerges:'
            w.toString().contains(g.toString())
            w.shape.hashCode()==g.shape.hashCode()
            w.translation().hashCode()==g.translation().hashCode()
            w.indicesMap().hashCode()==g.indicesMap().hashCode()
            w.spread().hashCode()==g.spread().hashCode()
            w.offset().hashCode()==g.offset().hashCode()

        where :
            gradient | expectedWeight
             -3      | 0.00299
             -3      | 0.00869
              2      | 0.01182
             -3      | 0.01764
              2      | 0.02088
              2      | 0.02179
             -4      | 0.02661
             -3      | 0.03395
             -3      | 0.04355
              2      | 0.050200001
    }


}
