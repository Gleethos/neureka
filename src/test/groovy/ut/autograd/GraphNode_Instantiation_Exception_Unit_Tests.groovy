package ut.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.backend.api.ExecutionCall
import neureka.backend.api.operations.AbstractOperation
import neureka.backend.api.Operation
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
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'GraphNode throws exception when trying to instantiate with the wrong context.'()
    {
        given : 'Arguments used to call the GraphNode constructor where the context variable contains the wrong type.'
            Function function = Mock(Function)
            Supplier<Tsr> supplier = () -> Mock(Tsr)
            Object context = new Integer(3)

        when : 'We try to instantiate a GraphNode...'
            new GraphNode(function, context, supplier)

        then : 'The constructor throws the expected error message.'
            def exception = thrown(IllegalArgumentException)
            exception.message == "The passed context object for the GraphNode constructor is of type 'java.lang.Integer'.\n" +
                    "A given context must either be a GraphLock instance or an ExecutionCall."

        and : 'The function has not been called.'
            0 * function.isDoingAD()
    }

    def 'GraphNode throws exception when trying to instantiate with Function argument being null.'()
    {
        given : 'Arguments used to call the GraphNode constructor where the Function variable is null.'
            Function function = null
            Supplier<Tsr> supplier = () -> Mock(Tsr)
            Object context = Mock(GraphLock)

        when : 'We try to instantiate a GraphNode...'
            new GraphNode(function, context, supplier)

        then : 'The constructor throws the expected error message.'
            def exception = thrown(IllegalArgumentException)
            exception.message == "Passed constructor argument of type Function must not be null!"
    }


    def 'GraphNode throws exception when payload is null.'()
    {
        given : 'Arguments used to call the GraphNode constructor where the payload supplier return null.'
            Function function = Mock(Function)
            Supplier<Tsr> supplier = () -> null
            Object context = Mock(GraphLock)

        when : 'We try to instantiate a GraphNode...'
            new GraphNode(function, context, supplier)

        then : 'The constructor throws the expected error message.'
            def exception = thrown(NullPointerException)
            exception.message == "The supplied payload Tsr must no be null!"
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
            1 * context.getTensors() >> inputs
            2 * inputsNodeMock.getLock() >> Mock( GraphLock )
            0 * function.isDoingAD() >> true
            0 * payload.getDevice() >> device
            0 * payload.to( _ )
            0 * device.cleaning( payload, _ )
            (1..2) * function.getOperation() >> type
            (0.._) * type.isDifferentiable() >> true
            (1.._) * type.isInline() >> true
            1 * inputs[0].get( GraphNode.class ) >> inputsNodeMock
            1 * inputs[1].get( GraphNode.class ) >> null
            0 * inputs[2].get( GraphNode.class ) >> null
            0 * context.allowsForward() >> true
    }


    def 'GraphNode instantiation throws exception because GraphNode instances of input tensors do not share the same GraphLock.'()
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
            def otherInputsNodeMock = Mock( GraphNode )

        when : 'We try to instantiate a GraphNode...'
            new GraphNode( function, context, supplier )

        then : 'The expected exception message is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message ==
                    "GraphNode instances found in input tensors do not share the same GraphLock instance.\n" +
                    "The given input tensors of a new node must be part of the same locked computation graph!"

        and : 'The mock objects are being called as expected.'
            1 * context.getTensors() >> inputs
            3 * inputsNodeMock.getLock() >> Mock( GraphLock )
            1 * otherInputsNodeMock.getLock() >> Mock( GraphLock )
            0 * function.isDoingAD() >> true
            0 * payload.getDevice() >> device
            0 * payload.to( _ )
            0 * device.cleaning( payload, _ )
            (1.._) * function.getOperation() >> type
            (0.._) * type.isDifferentiable() >> true
            (1.._) * type.isInline() >> true
            1 * inputs[0].get( GraphNode.class ) >> inputsNodeMock
            1 * inputs[1].get( GraphNode.class ) >> inputsNodeMock
            1 * inputs[2].get( GraphNode.class ) >> otherInputsNodeMock
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
            1 * context.getTensors() >> inputs
            2 * inputsNodeMock.getLock() >> Mock( GraphLock )
            0 * function.isDoingAD() >> true
            1 * type.isInline() >> true
            0 * payload.getDevice() >> device
            0 * payload.to( _ )
            0 * device.cleaning( payload, _ )
            1 * inputs[0].get( GraphNode.class ) >> inputsNodeMock
            0 * inputs[1].get( GraphNode.class ) >> inputsNodeMock
            0 * inputs[2].get( GraphNode.class ) >> inputsNodeMock
            0 * inputsNodeMock.getMode() >> -2
            1 * inputsNodeMock.usesAD() >> true
            0 * inputs[0].rqsGradient() >> true
            0 * inputs[1].rqsGradient() >> false
            0 * inputs[2].rqsGradient() >> true
            0 * context.allowsForward() >> true
            0 * context.allowsBackward() >> true
            (2..3) * function.getOperation() >> type
            (1..3) * type.getFunction() >> "SOME_TEST_FUNCTION_STRING"
            0 * type.getOperator() >> "*"
            0 * inputsNodeMock.getPayload() >> payload
            0 * payload.hashCode() >> 3

    }


}
