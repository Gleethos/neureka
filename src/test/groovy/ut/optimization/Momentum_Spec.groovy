package ut.optimization

import neureka.Neureka
import neureka.Tensor
import neureka.optimization.Optimizer
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Shared
import spock.lang.Specification
import spock.lang.Subject

@Narrative('''

    Momentum is an extension to the gradient descent optimization 
    algorithm that allows the search to build inertia in a direction 
    in the search space and overcome the oscillations of noisy 
    gradients and coast across flat spots of the search space.

''')
@Subject([Optimizer])
class Momentum_Spec extends Specification
{
    @Shared Tensor<Double> w = Tensor.of(0d)
    @Shared Optimizer<Double> o = Optimizer.Momentum.create(w)

    def setupSpec()
    {
        reportHeader """
                The code below assumes that for we
                have the following 2 variables setup
                throughout every data table iteration:
                ```
                    Tensor<?> w = Tensor.of(0d)
                    Optimizer<?> o = Optimizer.Momentum.create(w)        
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


    def 'Momentum optimizes according to expected inputs' (
            int gradient, double expectedWeight
    ) {
        given : 'A new scalar gradient tensor is being created.'
            Tensor g = Tensor.of(expectedWeight)
        and : 'The following input is being applied to the tensor (and internal optimizer)...'
            w.set( Tensor.of( (double) gradient ) )
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
