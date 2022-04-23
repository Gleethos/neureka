package ut.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.backend.api.ExecutionCall
import neureka.backend.api.algorithms.fun.AutoDiff
import neureka.backend.api.operations.AbstractOperation
import neureka.calculus.Function
import neureka.devices.Device
import neureka.view.TsrStringSettings
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
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    def 'GraphNode instantiation works as expected when the context argument is a GraphLock.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr           payload  = Mock( Tsr )
            Supplier<Tsr> supplier = () -> payload
            Function      function = Mock( Function )
            Object        context = Mock( GraphLock )
            Device        device = Mock( Device )
        and : 'We set the payload up so that it return the mocked device!'
            payload.getDevice() >> device

        when : 'We try to instantiate a GraphNode...'
            new GraphNode( function, context, supplier )

        then : 'The mock objects are being called as expected.'
            1 * function.isDoingAD() >> true
            1 * payload.set( _ )
            1 * device.access( _ ) >> Mock(Device.Access)

    }

    def 'GraphNode instantiation works as expected when the context argument is an ExecutionCall.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock( Tsr )
            Tsr[] inputs = new Tsr[]{ Mock(Tsr), Mock(Tsr), Mock(Tsr) }
            Supplier<Tsr> supplier = () -> payload
            AbstractOperation type = Mock( AbstractOperation )
            Function function = Mock( Function )
            Object call = Mock( ExecutionCall )
            Device device = Mock( Device )
            def inputsNodeMock = Mock( GraphNode )
            GraphNode result

        when : 'We try to instantiate a GraphNode...'
            result = new GraphNode( function, call, supplier )

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
            (3.._) * call.inputs() >> inputs
            (5.._) * inputsNodeMock.getLock() >> Mock( GraphLock )
            (1.._) * function.isDoingAD() >> true
            (1.._) * payload.getDevice() >> device
            (1.._) * payload.set( _ )
            (1.._) * device.access( _ ) >> Mock( Device.Access )
            (4.._) * inputs[0].getGraphNode() >> inputsNodeMock
            (3.._) * inputs[1].getGraphNode() >> inputsNodeMock
            (3.._) * inputs[2].getGraphNode() >> inputsNodeMock
            (1.._) * inputsNodeMock.getMode() >> -2
            (1.._) * inputs[0].rqsGradient() >> true
            (1.._) * inputs[1].rqsGradient() >> false
            (1.._) * inputs[2].rqsGradient() >> true
            (1.._) * call.autogradMode() >> AutoDiff.FORWARD_AND_BACKWARD
            (3.._) * function.getOperation() >> type
            (0.._) * type.isDifferentiable() >> true
            (3.._) * type.isInline() >> false
            (0.._) * type.getOperator() >> "*"
            (3.._) * inputsNodeMock.getPayload() >> payload
            (3.._) * payload.hashCode() >> 3

    }

}
