package it.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.JITProp
import neureka.calculus.assembly.FunctionBuilder
import spock.lang.Specification

class JITProp_Autograd_Tensor_Integration_Spec extends Specification
{

    def setupSpec()
    {
        reportHeader """
            <h2>Autograd Tensor Integration Tests</h2>
            <p>
                This specification contains tests which
                cover the autograd behavior of tensors. <br>
                The classes involved in governing the tested features are
                the Tsr, GraphNode and Function (& implementations) classes.
            </p>
        """

    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Test pending error optimization'()
    {
        given : 'The view settings are being set to legacy.'
            Neureka.get().settings().view().isUsingLegacyView = true
        and : 'The simple scalar tensors are being instantiated, where one requires gradients.'
            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when : 'The following calculations are being executed :'
            Tsr s =  (a*b) + 2
            Tsr x = s * (s+c)

            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(false)
            x.backward(Tsr.of(1))
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)

        then :
            c.toString().contains("(3.0):g:(-6.0)")
            a.toString().contains("(2.0):g:(36.0)")

        when :
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(false)
            x.backward(4)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)

        then :
            c.toString().contains("(3.0):g:(-6.0)")
            a.toString().contains("(2.0):g:(36.0)")
    }

    def 'Test JIT propagation variant one.'()
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr c = Tsr.of(3).setRqsGradient(true)

            Tsr s =  (a*b) + 2
            Tsr x = s * (s+c)

        when : x.backward(Tsr.of(1))
        then :
            c.toString().contains("g:(-6.0)")
            a.toString().contains("g:(null)")

        when : a.applyGradient()
        then :
            c.toString().contains("g:(-6.0)")
            a.toString().contains("(38.0):g:(null)")
    }



    def 'Test JIT propagation variant two.'()
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr c = Tsr.of(3).setRqsGradient(true)

            Tsr s =  (a*b) + 2 // -6 = (2*-4) +2
            Tsr x = s * (s+c) //  -6 * (-6+3) // 18

        when : x.backward(Tsr.of(1))

        then :
            c.toString().contains("g:(-6.0)")
            a.toString().contains("g:(null)")

        when :
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Tsr y = a+3 //JIT-prop will be activated here...
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)

        then :
            y.toString().contains("(41.0)")
            c.toString().contains("g:(-6.0)")
            a.toString().contains("(38.0):g:(null)")
    }



    def 'Gradient auto-apply kicks in when used AD uses JIT prop'()
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)

            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-3)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when :
            Tsr s = (a+b) * c // (2 - 3) * 3 = -3
            Tsr x = (s/a)+s // (-3)/2 -3 = -4.5

        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)
            !s.getGraphNode().isReliesOnJustInTimeProp()
            !a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            !c.getGraphNode().isReliesOnJustInTimeProp()

        when : x.backward(1)
        then :
            a.has(JITProp.class)
            !b.has(JITProp.class)
            c.has(JITProp.class)
            s.getGraphNode().isReliesOnJustInTimeProp()
            a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            c.getGraphNode().isReliesOnJustInTimeProp()
            a.toString().contains("g:(0.75)")
            c.toString().contains("g:(null)")
            x.toString().contains("(-4.5)")

        when :
            def f = new FunctionBuilder( Neureka.get().backend() ).build("I[0]*I[1]", false)
            Tsr[] inputs = new Tsr[]{c, a}
            Tsr result = f(inputs) // Should have no affect!

        then :
            s.getGraphNode().isReliesOnJustInTimeProp()
            a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            c.getGraphNode().isReliesOnJustInTimeProp()
            ! result.toString().contains("d[1]:")
            ! result.toString().contains("d[1]:")

            a.toString().contains("(2.0):g:(0.75)")
            c.toString().contains("g:(null)")
            x.toString().contains("(-4.5)")

        when :
            f = new FunctionBuilder( Neureka.get().backend() ).build("I[0]*I[1]", true)
            result = f(inputs) // Should trigger JIT

        then :
            result.toString().contains("d[1]:(7.25)")
            result.toString().contains("d[1]:(1.5)")

            a.toString().contains("(7.25):g:(null)")
            c.toString().contains("(1.5):g:(null)")// Both input values have been updated!
            x.toString().contains("(-4.5)")

            !c.has(JITProp.class)
            !b.has(JITProp.class)

            !s.getGraphNode().isReliesOnJustInTimeProp()
            !a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            !c.getGraphNode().isReliesOnJustInTimeProp()
    }


    def 'Test no preemptive gradient apply when not requested and auto apply and JIT_prop'() // This has been checked allot! :)
    {
        given :
            Neureka.get().settings().view().setIsUsingLegacyView(true)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(true)

            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-3)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when :
            Tsr s = (a+b) * c // (2 - 3) * 3 = -3
            Tsr x = (s/a) + s // (-3)/2 - 3 = -4.5

            System.gc() // Why? -> To test if even a clean graph will work properly!
            Thread.sleep(50)
            // f'(a) = 5.25 ! -> Checked! (Gradient = 5.25)
            // f'(c) = -1.5 ! -> Checked! (Gradient = -1.5)

        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)
            !s.has(JITProp.class)
            !s.getGraphNode().isReliesOnJustInTimeProp()
            !a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            !c.getGraphNode().isReliesOnJustInTimeProp()

        when : x.backward(1)
        then :
            s.getGraphNode().isReliesOnJustInTimeProp()
            a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            c.getGraphNode().isReliesOnJustInTimeProp()
            a.has(JITProp.class)
            !b.has(JITProp.class)
            c.has(JITProp.class)
            a.toString().contains("(2.0):g:(0.75)") // Partial gradient -> Later completed by JITProp
            c.toString().contains("(3.0):g:(null)") // Gradient created by JIT later on...
            x.toString().contains("(-4.5)")

        when :
            def f = new FunctionBuilder( Neureka.get().backend() ).build("I[0]*I[1]", false)
            Tsr[] inputs = new Tsr[]{c, a}
            Tsr result = f(inputs) // No changes to inputs! No derivatives!

        then :
            ! result.toString().contains("d[1]:")
            ! result.toString().contains("d[1]:")

            a.toString().contains("(2.0):g:(0.75)")
            c.toString().contains("(3.0):g:(null)")
            x.toString().contains("(-4.5)")

        when :
            f = new FunctionBuilder( Neureka.get().backend() ).build("I[0]*I[1]", true)
            result = f(inputs) // No changes to inputs, BUT derivatives!

        then :
            result.toString().contains("d[1]:(2.0)")
            result.toString().contains("d[1]:(3.0)")

            a.toString().contains("(2.0):g:(0.75)")
            c.toString().contains("(3.0):g:(null)")
            x.toString().contains("(-4.5)")

        when : a.setGradientApplyRequested(true)
        then : a.toString().contains("g:(0.75)")

        when : result = f(inputs) // Changes to inputs AND derivatives!
        then :
            result.toString().contains("d[1]:(7.25)")
            result.toString().contains("d[1]:(3.0)")

            a.toString().contains("(7.25):g:(null)") // Gradient of a has been applied! (5.25)
            c.toString().contains("(3.0):g:(-1.5)") // Final gradient for c (Checked!)
            x.toString().contains("(-4.5)")

            !a.has(JITProp.class)
            !c.has(JITProp.class)

            !s.getGraphNode().isReliesOnJustInTimeProp()
            !a.getGraphNode().isReliesOnJustInTimeProp()
            !b.getGraphNode().isReliesOnJustInTimeProp()
            !c.getGraphNode().isReliesOnJustInTimeProp()
    }



    def 'Test autograd without JIT and auto apply.'()
    {
        given :
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
            Neureka.get().settings().view().setIsUsingLegacyView(true)

            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-3)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when :
            Tsr s = (a+b) * c // (2 - 3) * 3 = -3
            Tsr x = (s/a)+s // (-3)^2 -3 = 6
        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)

        when : x.backward(1)
        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)
            a.toString().contains("g:(5.25)")// This has been checked!
            c.toString().contains("g:(-1.5)")// This has been checked!
            x.toString().contains("(-4.5)")

        when :
            a.applyGradient()
            c.applyGradient()
        then :
            a.toString().contains("(7.25):g:(null)")
            c.toString().contains("(1.5):g:(null)")
    }



    def 'Test in-differential and JIT with auto apply'()
    {
        given :
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
            Neureka.get().settings().view().setIsUsingLegacyView(true)

            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-3)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when :
            Tsr s = (a+b) * c // (2 - 3) * 3 = -3
            Tsr x = (s^a)+s // (-3)^2 -3 = 6

        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)

        when : x.backward(3)
        then :
            a.has(JITProp.class)
            !b.has(JITProp.class)
            c.has(JITProp.class)
            a.toString().contains("g:(NaN)")// NaN is expected! (derivative not possible!)
            c.toString().contains("g:(null)")
    }



    def 'Test no JIT prop when forward AD'()
    {
        given :
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
            Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
            Neureka.get().settings().view().setIsUsingLegacyView(true)

            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        when :
            Tsr s = (a+b) * c
            Tsr x = (s^2)+s

        then :
            s.toString().contains("->d[1]:(-2.0)")
            s.toString().contains("->d[1]:(3.0)")
            s.toString().contains("[1]:(-6.0)")
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)

        when : x.backward(3)
        then :
            !a.has(JITProp.class)
            !b.has(JITProp.class)
            !c.has(JITProp.class)
            a.toString().contains("g:(-99.0)")
            c.toString().contains("g:(66.0)")
    }


}
