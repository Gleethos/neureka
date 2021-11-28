package it.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.view.TsrStringSettings
import spock.lang.Specification

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

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.scientific( true )
            it.multiline( false )
            it.withGradient( true )
            it.withCellSize( 1 )
            it.withValue( true )
            it.withRecursiveGraph( false )
            it.withDerivatives( true )
            it.withShape( true )
            it.cellBound( false )
            it.withPostfix(  "" )
            it.withPrefix(  ""  )
            it.withSlimNumbers(  false )  
        })
    }

    def 'Simple automatic differentiation and propagation.'()
    {
        reportInfo """
            How can I compute gradients with Neureka automatically?
        """

        given : """
                Because (by default) Neureka is not too eager when it comes to backpropagation
                we have to set the following flags :
            """
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(false)

        and : 'We create a simple tensor and set rqsGradient to true in order to track dependent computation.'
            def x = Tsr.of([2, 2], 1).setRqsGradient(true)

        expect : 'The tensor should look as follows : '
            x.toString().contains("(2x2):[1.0, 1.0, 1.0, 1.0]")

        when : 'The following "+" operation is being applied...'
            def y = x + 2

        then : 'The new tensor now contains four threes.'
            y.toString().contains("(2x2):[3.0, 3.0, 3.0, 3.0]")

        and : 'Because "y" was created as a result of a default operation, it now has a graph node as component.'
            y.has( GraphNode.class )

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
            def someTensor = Tsr.newInstance()

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
                result.backward() is equivalent to out.backward(Tsr.of(1)).
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
            !result.has( GraphNode.class )

    }


}
