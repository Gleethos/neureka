package ut.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.backend.api.ExecutionCall
import neureka.backend.api.operations.AbstractOperation
import spock.lang.Specification

import java.util.function.Supplier

class GraphNode_Instantiation_Unit_Tests extends Specification
{

    def setupSpec()
    {
        reportHeader """
            <h2> GraphNode Instantiation </h2>
            <p>
                Specified below are strict tests covering the behavior
                of the GraphNode class during instantiation.
            </p>
        """
    }

    def setup() {
        Neureka.instance().reset()
    }

    def 'GraphNode instantiation works as expected when the context argument is a GraphLock.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock( Tsr )
            Supplier<Tsr> supplier = () -> payload
            Function function = Mock( Function )
            Object context = Mock( GraphLock )
            Device device = Mock( Device )
            payload.device() >> device

        when : 'We try to instantiate a GraphNode...'
            new GraphNode( function, context, supplier )

        then : 'The mock objects are being called as expected.'
            1 * function.isDoingAD() >> true
            1 * payload.set( _ )
            1 * device.cleaning( payload, _ )

    }

    def 'GraphNode instantiation works as expected when the context argument is an ExecutionCall.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock( Tsr )
            Tsr[] inputs = new Tsr[]{ Mock(Tsr), Mock(Tsr), Mock(Tsr) }
            Supplier<Tsr> supplier = () -> payload
        AbstractOperation type = Mock( AbstractOperation )
            Function function = Mock( Function )
            Object context = Mock( ExecutionCall )
            Device device = Mock( Device )
            def inputsNodeMock = Mock( GraphNode )
            GraphNode result

        when : 'We try to instantiate a GraphNode...'
            result = new GraphNode( function, context, supplier )

        then : 'The resulting GraphNode has expected properties.'
            result.getNodeID() != 0
            result.type() == "BRANCH"
            result.getLock() != null
            result.function != null
            result.hasDerivatives() == false
            result.usesAD()
            result.size() == 0
            result.usesReverseAD()

        and : 'The mock objects have been called as expected.'
            3 * context.getTensors() >> inputs
            5 * inputsNodeMock.getLock() >> Mock( GraphLock )
            1 * function.isDoingAD() >> true
            1 * payload.device() >> device
            1 * payload.set( _ )
            1 * device.cleaning( payload, _ )
            4 * inputs[0].find(GraphNode.class) >> inputsNodeMock
            3 * inputs[1].find(GraphNode.class) >> inputsNodeMock
            3 * inputs[2].find(GraphNode.class) >> inputsNodeMock
            1 * inputsNodeMock.getMode() >> -2
            1 * inputs[0].rqsGradient() >> true
            1 * inputs[1].rqsGradient() >> false
            1 * inputs[2].rqsGradient() >> true
            1 * context.allowsForward() >> true
            1 * context.allowsBackward() >> true
            4 * function.getOperation() >> type
            4 * type.isDifferentiable() >> true
            0 * type.getOperator() >> "*"
            3 * inputsNodeMock.getPayload() >> payload
            3 * payload.hashCode() >> 3

    }

}
