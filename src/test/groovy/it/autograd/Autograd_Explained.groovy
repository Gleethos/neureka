package it.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import spock.lang.Specification
import spock.lang.Unroll

class Autograd_Explained extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <b> Autograd - Automatic Differentiation </b>                                                           <br>
            <br>
            Central to all neural networks in Neureka is the autograd package.                                      <br>
            The autograd package provides automatic differentiation for all default operations on Tensors.          <br>
            Neureka is a define-by-run library, which means that your backpropagation is defined by how             <br>
            your code is run, and that every single iteration can be different.                                     <br>
            <br>
            The class neureka.Tsr is the central class of the main package.                                         <br>
            If you set its attribute rqsGradient to True, Neureka starts to track all operations on it.             <br>
            When you finish the forward pass of your network                                                        <br>
            you can call .backward() and have all the gradients computed                                            <br>
            and distributed to the tensors requiring them automatically.                                            <br>
            <br>
            <b> Tensors and Gradients </b>                                                                          <br>
            <br>                                                                                                        <br>
            The gradient for a tensor will be accumulated into a child tensor (component) which                     <br>
            can be accessed via the .getGradient() method.                                                          <br>
            <br>
            To stop a tensor from tracking history, you can call .detach() to detach it from the                    <br>
            computation history, and to prevent future computation from being tracked.                              <br>
            <br>                                         
        """
        reportHeader """
            There’s one more class which is very important for autograd implementation : the GraphNode.             <br>
            Tsr and GraphNode instances are interconnected and build up an acyclic graph,                           <br>       
            that encodes a complete history of computation.                                                         <br>                   
            Each tensor has a .getGraphNode() attribute that references the GraphNode                               <br>             
            that has created a given Tsr instance.                                                                  <br>
            (except for Tsr created by the user or created by a "detached" Function instance... ).                  <br>
            <br>
        """
        reportHeader """
            If you want to compute the derivatives, you can call .backward() on a Tensor.                           <br>
            If the given Tsr is a scalar (i.e. it holds one element and has shape "(1)"), you do not need to        <br>
            specify any arguments to backward(), however if it has more elements,                                   <br>
            you should specify a gradient argument that is a tensor of matching shape.                              <br>
        """
    }

    def 'Simple automatic differentiation and propagation.'()
    {
        reportInfo """
            How can I compute gradients with Neureka automatically?
        """

        given : 'Neureka is being reset in order to assure that configurations set to default.'
            Neureka.instance().reset()
        and : """
                Because (by default) Neureka is not too eager when it comes to backpropagation
                we have to set the following flags :
            """
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
            Neureka.instance().settings().autograd().setIsRetainingPendingErrorForJITProp(false)

        and : 'We create a simple tensor and set rqsGradient to true in order to track dependent computation.'
            def x = new Tsr([2, 2], 1).setRqsGradient(true)

        expect : 'The tensor should look as follows : '
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]")

        when : 'The following "+" operation is being applied...'
            def y = x + 2

        then : 'The new tensor now contains four threes.'
            y.toString().contains("(2x2):[3.0, 3.0, 3.0, 3.0]")

        and : 'Because "y" was created as a result of a default operation, it now has a graph node as component.'
            y.has(GraphNode.class)

        when : 'We do more computations on "y" ...'
            def z = y * y * 3

        then : 'As expected, this new tensor contains four times 27 :'
            z.toString().contains("(2x2):[27.0, 27.0, 27.0, 27.0]")

        when : """
                We call the "mean()" method as a simple loss function!
                This produces a scalar output tensor which is ideal as entrypoint
                for the autograd algorithm.
            """
            def result = z.mean()

        then : 'This "result" tensor will be the expected scalar :'
            result.toString().contains("(1x1):[27.0]")

        when : 'Any new tensor is created...'
            def someTensor = new Tsr()

        then : 'The autograd flag will always default to "false" :'
            someTensor.rqsGradient() == false

        when : 'We take a look at said property of the previously created "result" variable...'
            def resultRequiresGradient = z.rqsGradient()

        then : """
                We will notice that "result" does NOT require gradients!
                Although one of it's "ancestors" does require gradients (namely: "x"),
                this variable itself will not hold any gradients except for when it
                propagates them ...
            """
            resultRequiresGradient == false

        when : """
                We now try to backpropagate! Because "result" contains a single scalar,
                result.backward() is equivalent to out.backward(new Tsr(1)).
            """
            z.backward(0.25)

        then : """
                The tensor which requires gradients, namely "x" now has the expected gradients :
        """
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]:g:[4.5, 4.5, 4.5, 4.5]")

        when : 'We now try to access the gradient...'
            def gradient = x.getGradient()

        then : 'This given gradient is as expected !'
            gradient.toString() == "(2x2):[4.5, 4.5, 4.5, 4.5]"

        when : 'It is time to free some memory because our history of computation has grown a bit...'
            result.detach()

        then : 'Our latest tensor will now longer have a strong reference to a soon to be garbage collected past !'
            !result.has(GraphNode.class)

    }


    @Unroll
    def 'Advanced backpropagation on all AD-Modes'(
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

        given : 'Neureka is being reset in order to assure that configurations set to default.'
            Neureka.instance().reset()
        and :
            Neureka.instance().settings().autograd().isApplyingGradientWhenRequested = whenRsd
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed = whenUse
            Neureka.instance().settings().autograd().isRetainingPendingErrorForJITProp = doJIT

        and :
            def x = new Tsr([2, 2], 1).setRqsGradient(true)
            def y = x + 2
            Binding binding = new Binding()
            binding.setVariable('x', x)
            binding.setVariable('y', y)

        when : "The code snippet is being execute..."
            Tsr z = new GroovyShell(binding).evaluate((code))

        then : 'As expected, this new tensor contains four times 27 :'
            z.toString().contains("(2x2):[27.0, 27.0, 27.0, 27.0]")

        when : """
                We call the "mean()" method as a simple loss function!
                This produces a scalar output tensor which is ideal as entrypoint
                for the autograd algorithm.
            """
            def result = z.mean()

        then : 'This "result" tensor will be the expected scalar :'
            result.toString().contains("(1x1):[27.0]")

        when : """
                We now try to backpropagate! Because "result" contains a single scalar,
                result.backward() is equivalent to out.backward(new Tsr(1)).
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
            x.setGradientApplyRqd( true )
            xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterRqd )

        when :
            x * 2
            xAsStr = x.toString()

        then : 'The variable "x" contains every expected String :'
            xAsStr.matches( afterAll )

        when : 'It is time to free some memory because our history of computation has grown a bit...'
        result.detach()

        then : 'Our latest tensor will now longer have a strong reference to a soon to be garbage collected past !'
        !result.has(GraphNode.class)

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




}
