package ut.tensors

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

class Tensor_Gradient_Unit_Test extends Specification
{
    def setupSpec()
    {
        reportHeader """
                <h2> Tensor Gradient Unit Tests </h2>
                <br>
                <b>Why is there a difference between "rqsGradient()" and "hasGradient()" ? :</b>
                <br><br>
                <p>
                    The latter property simply tells if a tensor has another tensor as component.
                    This however does not necessitate it to also require gradients via the autograd system.
                    This is what the prior property is for.            
                </p>
            """
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Tensors can have gradients but not require them.'()
    {
        given : 'A new simple tensor.'
            Tsr t = new Tsr(-3)

        and : 'A second tensor viewed as gradient.'
            Tsr g = new Tsr(9)

        when : 'The gradient tensor is added to the prior tensor as component.'
            t.set( g )

        then : 'The prior tensor "hasGradient()" but does not "rqsGradient()"'
            t.has(Tsr.class)
            t.hasGradient()
            !t.rqsGradient()
    }


    def 'Tensors that have gradients but do not require them still print them.'()
    {
        given : 'A new simple tensor.'
            Tsr t = new Tsr(-3)

        and : 'A second tensor viewed as gradient.'
            Tsr g = new Tsr(9)

        when : 'The gradient tensor is added to the prior tensor as component.'
            t.set( g )

        then : 'The prior tensor will also include its gradient in the "toString()" result.'
            t.toString().contains("]:g:[")
    }


    def 'Gradient of tensor is being applies regardless of the tensor requiring gradient or not'(
        boolean requiresGradient, String expected
    ) {
        given : 'A new simple tensor.'
            Tsr t = new Tsr(-3)

        and : 'A second tensor viewed as gradient.'
            Tsr g = new Tsr(9)

        and : 'The gradient tensor is added to the prior tensor as component.'
            t.set( g )

        when : 'The request to apply the gradient is being made.'
            t.applyGradient()

        then : 'The tensor changed as expected.'
            t.toString().contains(expected)

        where :
            requiresGradient || expected
            true             || "(1):[6.0]"
            false            || "(1):[6.0]"
    }


}
