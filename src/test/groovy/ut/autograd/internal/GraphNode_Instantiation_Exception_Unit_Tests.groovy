package ut.autograd.internal

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.GraphNode
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.api.template.operations.AbstractOperation
import neureka.math.Function
import neureka.devices.Device
import neureka.view.NDPrintSettings
import spock.lang.Specification

import java.util.function.Supplier

class GraphNode_Instantiation_Exception_Unit_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> GraphNode Instantiation Tests </h2>
            <p>
                Specified below are strict tests covering the behavior
                of the GraphNode class during instantiation where
                inputs are setup to cause expected exceptions.
            </p>
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

    def 'GraphNode instantiation throws exception because tensors of ExecutionCall do not return GraphNode instances.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock( Tsr )
            Tsr[] inputs = new Tsr[]{ Mock(Tsr), Mock(Tsr), Mock(Tsr) }
            Supplier<Tsr> supplier = () -> payload
            Function function = Mock( Function )
            Operation type = Mock(Operation)
            Object context = Mock( ExecutionCall )
            Device device = Mock( Device )
            def inputsNodeMock = Mock( GraphNode )

        when : 'We try to instantiate a GraphNode...'
            new GraphNode( function, context, supplier )

        then : 'The expected exception message is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message ==
                    "Input tensor at index '1' did not return a GraphNode instance." +
                    "Input tensors of a new GraphNode must be part of the computation graph!"

        and : 'The mock objects are being called as expected.'
            1 * context.inputs() >> inputs
            0 * function.isDoingAD() >> true
            0 * payload.getDevice() >> device
            0 * payload.to( _ )
            0 * device.cleaning( payload, _ )
            (1..2) * function.getOperation() >> type
            (0.._) * type.isDifferentiable() >> true
            (1.._) * type.isInline() >> true
            1 * inputs[0].getGraphNode() >> Optional.of(inputsNodeMock)
            1 * inputs[1].getGraphNode() >> Optional.empty()
            0 * inputs[2].getGraphNode() >> Optional.empty()
            0 * context.allowsForward() >> true
    }

    def 'GraphNode throws an exception when trying to execute an inline operation on inputs with active autograd.'()
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

        when : 'We try to instantiate a GraphNode...'
            new GraphNode( function, context, supplier )

        then : 'The expected exception message is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message ==
                    "Trying to apply inline operation 'SOME_TEST_FUNCTION_STRING'\n"+
                    "on active autograd computation graph in non detached function.\n"+
                    "Please use detached functions instead! ( 'Function.create(\"SOME_TEST_FUNCTION_STRING(...)\", false)' )\n"

        and : 'The mock objects have been called as expected.'
            1 * context.inputs() >> inputs
            0 * function.isDoingAD() >> true
            1 * type.isInline() >> true
            0 * payload.getDevice() >> device
            0 * payload.to( _ )
            0 * device.cleaning( payload, _ )
            1 * inputs[0].getGraphNode() >> Optional.of(inputsNodeMock)
            0 * inputs[1].getGraphNode() >> Optional.of(inputsNodeMock)
            0 * inputs[2].getGraphNode() >> Optional.of(inputsNodeMock)
            0 * inputsNodeMock.getMode() >> -2
            1 * inputsNodeMock.usesAD() >> true
            0 * inputs[0].rqsGradient() >> true
            0 * inputs[1].rqsGradient() >> false
            0 * inputs[2].rqsGradient() >> true
            0 * context.allowsForward() >> true
            0 * context.allowsBackward() >> true
            (2..3) * function.getOperation() >> type
            (1..3) * type.getIdentifier() >> "SOME_TEST_FUNCTION_STRING"
            0 * type.getOperator() >> "*"
            0 * inputsNodeMock.getPayload() >> Optional.of(payload)
            0 * payload.hashCode() >> 3

    }


}
