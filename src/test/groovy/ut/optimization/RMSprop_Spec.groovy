package ut.optimization

import neureka.Neureka
import neureka.Tsr
import neureka.optimization.Optimizer
import neureka.optimization.implementations.RMSprop
import neureka.view.NDPrintSettings
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Subject

@Subject([Optimizer])
class RMSprop_Spec extends Specification
{
    @Shared Tsr<?> w = Tsr.of(0)
    @Shared Optimizer<?> o = new RMSprop<>(w)

    def setupSpec()
    {
        reportHeader """
                <h2> RMSprop Optimizer Behavior </h2>
                <br> 
                <p>
                    This specification check the behavior of the RMSprop class.        
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


    def 'RMSprop optimizes according to expected inputs' (
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
             -3      | 0.00316
             -3      | 0.00545
              2      | 0.00402
             -3      | 0.00586
              2      | 0.00466
              2      | 0.00349
             -4      | 0.00544
             -3      | 0.00682
             -3      | 0.00815
              2      | 0.00725
    }

}