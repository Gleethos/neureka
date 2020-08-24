package ut.tensors

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

class Tensor_Gradient_Unit_Test extends Specification
{

    /**
     *  Why is there a difference between "rqsGradient()" and "hasGradient()" ? :
     *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     *  The latter property simply tells if a tensor has another tensor as component.
     *  This however does not necessitate it to also require gradients via the autograd system.
     *  This is what the prior property is for.
     */
    def 'Tensors can have gradients but not require them.'()
    {
        given : 'The Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A new simple tensor.'
            Tsr t = new Tsr(-3)

        and : 'A second tensor viewed as gradient.'
            Tsr g = new Tsr(9)

        when : 'The gradient tensor is added to the prior tensor as component.'
            t.add( g )

        then : 'The prior tensor "hasGradient()" but does not "rqsGradient()"'
            t.has(Tsr.class)
            t.hasGradient()
            !t.rqsGradient()
    }


    def 'Tensors that have gradients but do not require them still print them.'()
    {
        given : 'The Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A new simple tensor.'
            Tsr t = new Tsr(-3)

        and : 'A second tensor viewed as gradient.'
            Tsr g = new Tsr(9)

        when : 'The gradient tensor is added to the prior tensor as component.'
            t.add( g )

        then : 'The prior tensor will also include its gradient in the "toString()" result.'
            t.toString().contains("]:g:[")
    }


}
