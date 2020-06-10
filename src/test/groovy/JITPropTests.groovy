import neureka.Neureka
import neureka.Tsr
import neureka.autograd.JITProp
import neureka.calculus.factory.assembly.FunctionBuilder
import org.junit.Test

class JITPropTests {

    @Test
    void test_auto_apply_when_used_on_AD_function_an_JIT_prop()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setIsUsingLegacyView(true)
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(false)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s/a)+s // (-3)/2 -3 = -4.5

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert !s.graphNode().reliesOnJustInTimeProp()
        assert !a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert !c.graphNode().reliesOnJustInTimeProp()
        x.backward(1)
        assert a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert c.has(JITProp.class)
        assert s.graphNode().reliesOnJustInTimeProp()
        assert a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert c.graphNode().reliesOnJustInTimeProp()
        assert a.toString().contains("g:(0.75)")
        assert c.toString().contains("g:(null)")
        assert x.toString().contains("(-4.5)")

        def f = FunctionBuilder.build("I[0]*I[1]", false)
        Tsr[] inputs = new Tsr[]{c, a}

        Tsr result = f(inputs) // Should have no affect!
        assert s.graphNode().reliesOnJustInTimeProp()
        assert a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert c.graphNode().reliesOnJustInTimeProp()
        assert ! result.toString().contains("d[1]:")
        assert ! result.toString().contains("d[1]:")

        assert a.toString().contains("(2.0):g:(0.75)")
        assert c.toString().contains("g:(null)")
        assert x.toString().contains("(-4.5)")

        f = FunctionBuilder.build("I[0]*I[1]", true)
        result = f(inputs) // Should trigger JIT
        assert result.toString().contains("d[1]:(7.25)")
        assert result.toString().contains("d[1]:(1.5)")

        assert a.toString().contains("(7.25):g:(null)")
        assert c.toString().contains("(1.5):g:(null)")// Both input values have been updated!
        assert x.toString().contains("(-4.5)")

        assert !c.has(JITProp.class)
        assert !b.has(JITProp.class)

        assert !s.graphNode().reliesOnJustInTimeProp()
        assert !a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert !c.graphNode().reliesOnJustInTimeProp()
        Neureka.instance().reset()
    }

    @Test
    void test_no_preemptive_apply_when_not_requested_and_auto_apply_and_JIT_prop() // Checked
    {
        Neureka.instance().reset()
        Neureka.instance().settings().view().setIsUsingLegacyView(true)
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s/a) + s // (-3)/2 - 3 = -4.5

        System.gc() // Why? -> To test if even a clean graph will work properly!
        Thread.sleep(50)
        // f'(a) = 5.25 ! -> Checked! (Gradient = 5.25)
        // f'(c) = -1.5 ! -> Checked! (Gradient = -1.5)

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert !s.has(JITProp.class)
        assert !s.graphNode().reliesOnJustInTimeProp()
        assert !a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert !c.graphNode().reliesOnJustInTimeProp()
        x.backward(1)
        assert s.graphNode().reliesOnJustInTimeProp()
        assert a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert c.graphNode().reliesOnJustInTimeProp()
        assert a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert c.has(JITProp.class)
        assert a.toString().contains("(2.0):g:(0.75)") // Partial gradient -> Later completed by JITProp
        assert c.toString().contains("(3.0):g:(null)") // Gradient created by JIT later on...
        assert x.toString().contains("(-4.5)")

        def f = FunctionBuilder.build("I[0]*I[1]", false)
        Tsr[] inputs = new Tsr[]{c, a}

        Tsr result = f(inputs) // No changes to inputs! No derivatives!
        assert ! result.toString().contains("d[1]:")
        assert ! result.toString().contains("d[1]:")

        assert a.toString().contains("(2.0):g:(0.75)")
        assert c.toString().contains("(3.0):g:(null)")
        assert x.toString().contains("(-4.5)")

        f = FunctionBuilder.build("I[0]*I[1]", true)
        result = f(inputs) // No changes to inputs, BUT derivatives!

        assert result.toString().contains("d[1]:(2.0)")
        assert result.toString().contains("d[1]:(3.0)")

        assert a.toString().contains("(2.0):g:(0.75)")
        assert c.toString().contains("(3.0):g:(null)")
        assert x.toString().contains("(-4.5)")

        a.setGradientApplyRqd(true)
        assert a.toString().contains("g:(0.75)")

        result = f(inputs) // Changes to inputs AND derivatives!
        assert result.toString().contains("d[1]:(7.25)")
        assert result.toString().contains("d[1]:(3.0)")

        assert a.toString().contains("(7.25):g:(null)") // Gradient of a has been applied! (5.25)
        assert c.toString().contains("(3.0):g:(-1.5)") // Final gradient for c (Checked)
        assert x.toString().contains("(-4.5)")

        assert !a.has(JITProp.class)
        assert !c.has(JITProp.class)

        assert !s.graphNode().reliesOnJustInTimeProp()
        assert !a.graphNode().reliesOnJustInTimeProp()
        assert !b.graphNode().reliesOnJustInTimeProp()
        assert !c.graphNode().reliesOnJustInTimeProp()
        Neureka.instance().reset()
    }

    @Test
    void test_autograd_without_JIT_and_auto_apply()
    {
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(false)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        Neureka.instance().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setIsUsingLegacyView(true);

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s/a)+s // (-3)^2 -3 = 6

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(1)
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert a.toString().contains("g:(5.25)")// This has been checked!
        assert c.toString().contains("g:(-1.5)")// This has been checked!
        assert x.toString().contains("(-4.5)")
        a.applyGradient()
        c.applyGradient()
        assert a.toString().contains("(7.25):g:(null)")
        assert c.toString().contains("(1.5):g:(null)")
        Neureka.instance().reset()
    }


    @Test
    void test_indifferential_and_JIT_with_auto_apply()
    {
        Neureka.instance().reset()
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setIsUsingLegacyView(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-3)
        Tsr c = new Tsr(3).setRqsGradient(true)

        Tsr s = (a+b) * c // (2 - 3) * 3 = -3
        Tsr x = (s^a)+s // (-3)^2 -3 = 6

        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(3)
        assert a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert c.has(JITProp.class)
        assert a.toString().contains("g:(NaN)")// NaN is expected! (derivative not possible!)
        assert c.toString().contains("g:(null)")
        Neureka.instance().reset()
    }

    @Test
    void test_no_JIT_prop_when_forward_AD(){
        Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(true)
        Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(true)
        Neureka.instance().settings().debug().setIsKeepingDerivativeTargetPayloads(false)
        Neureka.instance().settings().view().setIsUsingLegacyView(true)

        Tsr a = new Tsr(2).setRqsGradient(true)
        Tsr b = new Tsr(-4)
        Tsr c = new Tsr(3).setRqsGradient(true)
        Tsr s = (a+b) * c
        Tsr x = (s^2)+s
        assert s.toString().contains("->d[1]:(-2.0)")
        assert s.toString().contains("->d[1]:(3.0)")
        assert s.toString().contains("[1]:(-6.0)")
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        x.backward(3)
        assert !a.has(JITProp.class)
        assert !b.has(JITProp.class)
        assert !c.has(JITProp.class)
        assert a.toString().contains("g:(-99.0)")
        assert c.toString().contains("g:(66.0)")
        Neureka.instance().reset()
    }

}
