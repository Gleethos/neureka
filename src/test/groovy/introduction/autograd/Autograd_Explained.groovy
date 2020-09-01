package introduction.autograd

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
            Central to all neural networks in Neureka is the autograd package.                                      <br>
            The autograd package provides automatic differentiation for all default operations on Tensors.          <br>
            Neureka is a define-by-run library, which means that your backpropagation is defined by how             <br>
            your code is run, and that every single iteration can be different.                                     <br>
            <br>
        """
        reportHeader """
            The class neureka.Tsr is the central class of the main package.                                         <br>
            If you set its attribute rqsGradient to True, Neureka starts to track all operations on it.             <br>
            When you finish the forward pass of your network                                                        <br>
            you can call .backward() and have all the gradients computed                                            <br>
            and distributed to the tensors requiring them automatically.                                            <br>
            <br>
        """
        reportHeader """
            The gradient for a tensor will be accumulated into a child tensor (component) which                     <br>
            can be accessed via the .getGradient() method.                                                          <br>
        """
        reportHeader """
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

    def 'Automatic differentiation and propagation.'()
    {
        reportInfo """
            How can I compute gradients with Neureka automatically?
        """

        given : 'Neureka is being reset in order to assure that configurations set to default.'
            Neureka.instance().reset()

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
                The tensor which requires gradients, namely "x" doesn't have them! Why?!?
        """
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]:g:[null]")

        and : """
                Neureka is by default not too eager when it comes to backpropagation.
                This is because of the following three default settings being true :
        """
            Neureka.instance().settings().autograd().isApplyingGradientWhenRequested()
            Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()
            Neureka.instance().settings().autograd().isRetainingPendingErrorForJITProp()

        and : """
                It detects edge cases within the computation graph and
                simply saves pending error values for further
                accumulation.
        """
            y.getGraphNode().getPendingError().getToBeReceived() == 0

        when :
            x * 2

        then :
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]:g:[null]")

        when :
            x.setGradientApplyRqd( true )

        then :
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]:g:[null]")

        when :
            x * 2

        then :
            x.toString().contains("(2x2):[5.5, 5.5, 5.5, 5.5]:g:[null]")


    }






    @Unroll
    def 'Test all AD-Modes'(
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
            How can I compute gradients with Neureka automatically?
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

        when : "The code snippet '${code}' is being execute..."
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
