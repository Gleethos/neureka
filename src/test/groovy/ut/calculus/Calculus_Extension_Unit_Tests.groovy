package ut.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.autograd.ADAgent
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.backend.operations.OperationType
import neureka.calculus.backend.implementations.OperationTypeImplementation
import neureka.calculus.frontend.implementations.FunctionNode
import neureka.calculus.frontend.implementations.FunctionInput
import spock.lang.Specification

class Calculus_Extension_Unit_Tests extends Specification
{

    def 'Mock operation interacts with FunctionNode (AbstractFunction) instance as expected.'(){

        given : 'Neureka is being reset.'
            Neureka.instance().reset()

        and : 'A new mock operation type is being created.'
            def type = Mock(OperationType)

        and : 'A list of mocked function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output'
            Tsr output = Mock(Tsr)

        and : 'A mocked operation implementation.'
            def implementation = Mock(OperationTypeImplementation)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, false)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.getArity() >> 2
            function.isFlat()
            !function.doesAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call(new Tsr[0])

        then : 'The custom call hook should be accessed as outlined below.'
            (1.._) * type.implementationOf(_) >> implementation
            (1.._) * implementation.handleInsteadOfDevice(_,_) >> output

        and : 'The mocked output tensor never returns the mock device because our custom call hook replaces execution.'
            0 * output.device() >> Mock(Device)

        and : 'The ADAnalyzer of the mock implementation will not be called because "doAD" is set to "false".'
            0 * implementation.isImplementationSuitableFor(_)

        and : 'The agent creator is never accessed because "doAD" is set to false.'
            0 * implementation.supplyADAgentFor(_,_,_)

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output

    }

    def 'Lambda properties of mock implementation interact with FunctionNode (AbstractFunction) as expected.'()
    {
        given : 'Neureka is being reset.'
            Neureka.instance().reset()

        and : 'A mock agent.'
            def agent = Mock(ADAgent)

        and : 'A new operation type with a new implementation.'
            def type = Mock(OperationType)

        and : 'A list of function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output, an input and a graph node.'
            Tsr output = Mock(Tsr)
            Tsr input = Mock(Tsr)
            GraphNode node = Mock(GraphNode)

        and : 'A mocked operation implementation.'
            def implementation = Mock(OperationTypeImplementation)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, true)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.getArity() >> 2
            function.isFlat()
            function.doesAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call([input])

        then : 'The custom call hook is being accessed as outlined below.'
            (1.._) * type.isDifferentiable() >> true
            (1.._) * type.implementationOf(_) >> implementation
            (1.._) * implementation.handleInsteadOfDevice(_,_) >> output

        and : 'The GraphNode instance which will be created as tensor component interacts as follows.'
            (1.._) * input.find(GraphNode.class) >> node
            (1.._) * node.lock() >> Mock(GraphLock)
            (1.._) * input.device() >> Mock(Device) // Device is being queried for execution...
            _ * type.getOperator() >> 'test_identifier'
            (1.._) * output.device() >> Mock(Device)

        and : 'The given ADAnalyzer instance is being called because auto-differentiation is enabled.'
            (1.._) * input.rqsGradient() >> true
            (1.._) * implementation.canImplementationPerformForwardADFor(_) >> false
            (1.._) * implementation.canImplementationPerformBackwardADFor(_) >> true
            (1.._) * node.getPayload() >> input
            (1.._) * node.usesAD() >> true

        and : 'The agent creator is being accessed because "doAD" is set to true and the input requires gradients.'
            1 * implementation.supplyADAgentFor(_,_,_) >> agent
            1 * agent.derivative() >> null

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output

    }



}
