package ut.autograd

import neureka.Neureka
import neureka.Tensor
import neureka.autograd.GraphNode
import neureka.view.NDPrintSettings
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Unroll


@Subject([Neureka, Neureka.Settings, Neureka.Settings.AutoGrad])
class Autograd_Flags_Explained extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <b> Autograd Advanced - Custom Autograd </b>                                                            <br>
                                                                                                                    <br>
            Neureka does not necessarily perform autograd eagerly.                                                  <br>
            If required then auto-differentiation will occur as one would expect                                    <br>
            similarly to the way PyTorch's autograd works.                                                          <br>
            However for many use cases it might make sense to use different variants                                <br>
            of auto-differentiation.                                                                                <br>
            This specification covers precisely these different autograd modes.                                     <br>
                                                                                                                    <br>
        """
    }

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



    @Unroll
    def 'Advanced backpropagation on all AD-Modes '(
            String code,
            boolean whenRsd,
            boolean whenUse,
            boolean doJIT,
            String afterBack,
            String afterUse,
            String afterRqd,
            String afterAll
    ) {
        reportInfo """
            What is JIT-Prop and how does it affect autograd ? <b> Let's take a look ! </b>                 <br>
            This run covers the feature having the following settings :                                     <br>
                                                                                                            <br>
            <b> Neureka.instance().settings().autograd().: </b>                                             <br>
            isRetainingPendingErrorForJITProp := ${doJIT}                                                   <br>      
            isApplyingGradientWhenTensorIsUsed := ${whenUse}                                                <br>    
            isApplyingGradientWhenRequested := ${whenRsd}                                                   <br>  
                                                                                                            <br>
            ...code producing the result : '${code}'                                                        <br>                                                                                                        
                                                                                                            <br>                                                                                                       
            <b> is-Retaining-Pending-Error-For-JITProp : </b>                                               <br>
                                                                                                            <br>
            This flag enables an optimization technique which only propagates error values to               <br>
            gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them        <br>
            at divergent differentiation paths within the computation graph.                                <br>
            If the flag is set to true                                                                      <br>
            then error values will accumulate at such junction nodes.                                       <br>
            This technique however uses more memory but will                                                <br>
            improve performance for some networks substantially.                                            <br>
            The technique is termed JIT-Propagation.                                                        <br>
                                                                                                            <br>
                                                                                                            <br>
            <b> is-Applying-Gradient-When-Tensor-Is-Used : </b>                                             <br>
                                                                                                            <br>
            Gradients will automatically be applied (or JITed) to tensors as soon as                        <br>
            they are being used for calculation (GraphNode instantiation).                                  <br>
            This feature works well with JIT-Propagation.                                                   <br>
                                                                                                            <br>
                                                                                                            <br>      
            <b> is-Applying-Gradient-When-Requested : </b>                                                  <br>
                                                                                                            <br>
            Gradients will only be applied if requested.                                                    <br>
            Usually this happens immediately, however                                                       <br>
            if the flag 'applyGradientWhenTensorIsUsed' is set                                              <br>
            to true, then the tensor will only be updated by its                                            <br>
            gradient if requested AND the tensor is used fo calculation! (GraphNode instantiation).         <br>
                                                                                                            <br>
            <b> Let's take a look :  </b>
        """

        given : 'We configure Neureka autograd:'
            Neureka.get().settings().autograd().isApplyingGradientWhenRequested = whenRsd
            Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed = whenUse
            Neureka.get().settings().autograd().isRetainingPendingErrorForJITProp = doJIT

        and :
            def x = Tensor.of([2, 2], 1d).setRqsGradient(true)
            def y = x + 2
            Binding binding = new Binding()
            binding.setVariable('x', x)
            binding.setVariable('y', y)

        when : "The code snippet is being execute..."
        Tensor z = new GroovyShell(binding).evaluate((code))

        then : 'As expected, this new tensor contains four times 27 :'
            z.toString().contains("(2x2):[27.0, 27.0, 27.0, 27.0]")

        when : """
                We call the "mean()" method as a simple loss function!
                This produces a scalar output tensor which is ideal as entrypoint
                for the autograd algorithm.
            """
            def result = z.mean()

        then : 'This "result" tensor will be the expected scalar :'
            result.toString().contains("27.0")

        when : """
                We now try to backpropagate! Because "result" contains a single scalar,
                result.backward() is equivalent to out.backward(Tensor.of(1)).
            """
            z.backward(0.25)
            def xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterBack )

        when :
            x * 2
            xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterUse )

        when :
            x.setGradientApplyRequested( true )
            xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterRqd )

        when :
            x * 2
            xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterAll )

        when : 'It is time to free some memory because our history of computation has grown a bit...'
            result.mut.detach()

        then : 'Our latest tensor will now longer have a strong reference to a soon to be garbage collected past !'
            !result.has( GraphNode.class )

        where :
            code   | whenRsd | whenUse | doJIT || afterBack     | afterUse        | afterRqd        | afterAll
            'y*y*3'| false   | false   | false ||".*1.*4\\.5.*" |".*1.*4\\.5.*"   |".*1.*4\\.5.*"   |".*1.*4\\.5.*"
            'y*y*3'| true    | false   | false ||".*1.*4\\.5.*" |".*1.*4\\.5.*"   |".*5\\.5.*null.*"|".*5\\.5.*null.*"
            'y*y*3'| false   | true    | false ||".*1.*4\\.5.*" |".*5\\.5.*null.*"|".*5\\.5.*null.*"|".*5\\.5.*null.*"
            'y*y*3'| true    | true    | false ||".*1.*4\\.5.*" |".*1.*4\\.5.*"   |".*1.*4\\.5.*"   |".*5\\.5.*null.*"
            'y*y*3'| false   | false   | true  ||".*1.*null.*"  |".*1.*null.*"    |".*1.*null.*"    |".*1.*null.*"
            'y*y*3'| true    | false   | true  ||".*1.*null.*"  |".*1.*null.*"    |".*5\\.5.*null.*"|".*5\\.5.*null.*"
            'y*y*3'| false   | true    | true  ||".*1.*null.*"  |".*5\\.5.*null.*"|".*5\\.5.*null.*"|".*5\\.5.*null.*"
            'y*y*3'| true    | true    | true  ||".*1.*null.*"  |".*1.*null.*"    |".*1.*null.*"    |".*5\\.5.*null.*"

    }

    def 'We can create a shallow copy of a tensor detached from the computation graph.'()
    {
        given :
            var a = Tensor.ofFloats().withShape(2).andFill(-3, 1).setRqsGradient(true)
        when :
            var b = a * 2
            var c = b.detached()

        then :
            c !== b
            c === c.detached() // c is already detached, so this is a no-op
        and :
            b.isBranch()
            !c.isBranch()
        and :
            b.has(GraphNode)
            !c.has(GraphNode)
    }

}