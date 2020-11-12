package ut.autograd

import neureka.Tsr
import neureka.autograd.GraphNode
import org.slf4j.Logger
import spock.lang.Shared
import spock.lang.Specification

class GraphNode_Tensor_Exception_Unit_Tests extends Specification
{

    @Shared def oldLogger

    def setup() {
        oldLogger = Tsr._LOGGER
        Tsr._LOGGER = Mock( Logger )
    }

    def cleanup() {
        Tsr._LOGGER = oldLogger
    }


    def 'A tensor cannot be deleted if it is part of a graph and the tensor is used as derivative.'()
    {
        given : 'A new simple scalar tensor instance.'
            Tsr t = new Tsr( 1 )
        and : 'A GraphNode mock object which is being added to the tensor.'
            def node = Mock( GraphNode )
            t.set( node )

        when : 'We try to delete the tensor...'
            t.delete()

        then : 'The graph node object will return true for "isUsedAsDerivative()" inside the tensor...'
            1 * node.isUsedAsDerivative() >> true
        and : '...an exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == "Cannot delete a tensor which is used as derivative by the AD computation graph!"
        and : 'The exception message is also being thrown.'
            Tsr._LOGGER.error( "Cannot delete a tensor which is used as derivative by the AD computation graph!" )
    }

}
