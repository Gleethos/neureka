package ut.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Autograd - Automatic Differentiation")
@Narrative('''

    Central to all neural networks in Neureka is the autograd package.                                      
    The autograd package provides automatic differentiation for all default operations on Tensors.          
    Neureka is a define-by-run library, which means that your backpropagation is defined by how             
    your code is run, and that every single iteration can be different.                                     
                                                                                                            
    The class neureka.Tsr is the central class of the main package.                                         
    If you set its attribute 'rqsGradient' to True, Neureka starts to track all operations on it.           
    When you finish the forward pass of your network                                                        
    you can call .backward() and have all the gradients computed                                            
    and distributed to the tensors requiring them automatically.                                            
                                                                                                            
    The gradient for a tensor will be accumulated into a child tensor (component) which                     
    can be accessed via the '.getGradient()' method.                                                        
                                                                                                            
    To stop a tensor from tracking history, you can call '.detach()' to detach it from the                  
    computation history, and to prevent future computation from being tracked.     
            
''')
@Subject([Tsr, GraphNode])
class Autograd_Explained extends Specification
{
    def setupSpec()
    {
        reportHeader """
            There’s one more class which is very important for autograd implementation : the 'GraphNode class'!     
            Tsr and GraphNode instances are interconnected and build up an acyclic graph,                              
            that encodes a complete history of computation.                                                                        
            Each tensor has a .getGraphNode() attribute that references the GraphNode                                        
            that has created a given Tsr instance.                                                                  
            (except for Tsr created by the user or created by a "detached" Function instance... ).                  
           
        """
        reportHeader """
            If you want to compute the derivatives, you can call .backward() on a Tensor.                           
            If the given Tsr is a scalar (i.e. it holds one element and has shape "(1)"), you do not need to        
            specify any arguments to backward(), however if it has more elements,                                   
            you should specify a gradient argument that is a tensor of matching shape.                              
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

    def 'Simple automatic differentiation and propagation.'()
    {
        reportInfo """
            How can I compute gradients with Neureka automatically?
        """

        given : """
                The following flag states enable regular auto-grad (should also be the default):
            """
            Neureka.get().settings().autograd().setIsApplyingGradientWhenRequested(false)
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
            Neureka.get().settings().autograd().setIsRetainingPendingErrorForJITProp(false)

        and : 'We create a simple tensor and set rqsGradient to true in order to track dependent computation.'
            def x = Tsr.of([2, 2], 1d).setRqsGradient(true)

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
            result.unsafe.detach()

        then : 'Our latest tensor will now longer have a strong reference to a soon to be garbage collected past !'
            !result.has( GraphNode.class )

    }


}
