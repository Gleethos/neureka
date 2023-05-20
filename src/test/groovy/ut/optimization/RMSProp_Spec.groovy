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

    **Root Mean Squared Propagation**, or RMSProp, is an extension of gradient 
    descent and the AdaGrad version of gradient descent that uses a 
    decaying average of partial gradients in the adaptation of the 
    step size for each parameter.

''')
@Subject([Optimizer])
class RMSProp_Spec extends Specification
{
    @Shared Tensor<?> w = Tensor.of(0d)
    @Shared Optimizer<?> o = Optimizer.RMSProp.create(w)

    def setupSpec()
    {
        reportHeader """
                The code below assumes that for we
                have the following 2 variables setup
                throughout every data table iteration:
                ```
                    Tensor<?> w = Tensor.of(0d)
                    Optimizer<?> o = Optimizer.RMSProp.create(w)            
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


    def 'RMSprop optimizes according to expected inputs' (
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