package ut.autograd

import neureka.Tsr
import neureka.acceleration.Device
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.calculus.environment.ExecutionCall
import spock.lang.Specification

import java.util.function.Supplier

class GraphNode_Exception_Unit_Tests extends Specification
{

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
            0 * function.doesAD()
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
            2 * inputsNodeMock.lock() >> Mock( GraphLock )
            0 * function.doesAD() >> true
            0 * payload.device() >> device
            0 * payload.add( _ )
            0 * device.cleaning( payload, _ )
            1 * inputs[0].find(GraphNode.class) >> inputsNodeMock
            1 * inputs[1].find(GraphNode.class) >> null
            0 * inputs[2].find(GraphNode.class) >> null
            0 * context.allowsForward() >> true
    }


    def 'GraphNode instantiation throws exception because GraphNode instances of input tensors do not share the same GraphLock.'()
    {
        given : 'Mocked arguments used to call the GraphNode constructor.'
            Tsr payload = Mock( Tsr )
            Tsr[] inputs = new Tsr[]{ Mock(Tsr), Mock(Tsr), Mock(Tsr) }
            Supplier<Tsr> supplier = () -> payload
            Function function = Mock( Function )
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
            3 * inputsNodeMock.lock() >> Mock( GraphLock )
            1 * otherInputsNodeMock.lock() >> Mock( GraphLock )
            0 * function.doesAD() >> true
            0 * payload.device() >> device
            0 * payload.add( _ )
            0 * device.cleaning( payload, _ )
            1 * inputs[0].find(GraphNode.class) >> inputsNodeMock
            1 * inputs[1].find(GraphNode.class) >> inputsNodeMock
            1 * inputs[2].find(GraphNode.class) >> otherInputsNodeMock
            0 * context.allowsForward() >> true
    }

}
