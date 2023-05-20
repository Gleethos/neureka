package ut.autograd.internal

import neureka.Neureka
import neureka.Tensor
import neureka.autograd.GraphNode
import spock.lang.Shared
import spock.lang.Specification

class GraphNode_Tensor_Exception_Unit_Tests extends Specification
{

    @Shared def oldStream

    def setupSpec()
    {
        reportHeader """
            <h2> GraphNode Tensor Exceptions </h2>
            <p>
                Specified below are strict tests covering the exception behavior
                of the GraphNode class interacting with Tensor instances.
            </p>
        """
    }

    def setup() {
        Neureka.get().reset()
        oldStream = System.err
        System.err = Mock(PrintStream)
    }

    def cleanup() {
        System.err = oldStream
    }


    def 'A tensor cannot be deleted if it is part of a graph and the tensor is used as derivative.'()
    {
        given : 'A new simple scalar tensor instance.'
        Tensor<Double> t = Tensor.of( 1d )
        and : 'A GraphNode mock object which is being added to the tensor.'
            def node = Mock( GraphNode )

        when :
            t.set( node )
        then :
            1 * node.update(_) >> true

        when : 'We try to delete the tensor...'
            t.mut.delete()

        then : 'The graph node will not allow deletion...'
            1 * node.canBeDeleted() >> false // If it has ancestors it must not be deleted!
        and : '...an exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == "Cannot delete a tensor which is used as derivative by the AD computation graph!"
        and : 'The exception message is also being thrown.'
            1 * System.err.println({it.endsWith("Cannot delete a tensor which is used as derivative by the AD computation graph!")})
    }

}
