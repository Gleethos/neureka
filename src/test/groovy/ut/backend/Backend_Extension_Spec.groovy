package ut.backend

import neureka.Neureka
import neureka.Tsr
import neureka.devices.Device
import neureka.autograd.ADAgent
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.backend.api.Operation
import neureka.backend.api.Algorithm
import neureka.calculus.implementations.FunctionNode
import neureka.calculus.implementations.FunctionInput
import neureka.ndim.AbstractTensor
import neureka.ndim.config.NDConfiguration
import spock.lang.Specification

class Backend_Extension_Spec extends Specification
{
    def setupSpec()
    {
        Neureka.get().reset()

        reportHeader """
                   This specification defines the behavior of
                   Operation instances and their ability to be extended! <br> 
        """
    }

    def 'Mock operation interacts with FunctionNode (AbstractFunction) instance as expected.'() {

        given : 'A new mock operation type is being created.'
            def type = Mock(Operation)

        and : 'A list of mocked function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output'
            Tsr output = Mock(Tsr)
            var mutate = Mock(AbstractTensor.Unsafe)

        and : 'A mocked operation implementation.'
            def implementation = Mock(Algorithm)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, false)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.getArity() >> 2
            function.isFlat()
            !function.isDoingAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call(new Tsr[0])

        then : 'The custom call hook should be accessed as outlined below.'
            (1.._) * type.getAlgorithmFor(_) >> implementation
            (1.._) * implementation.dispatch(_,_) >> output
            (1.._) * output.getUnsafe() >> mutate
            (1.._) * mutate.setIsIntermediate(false) >> output
            (1.._) * output.isIntermediate() >> true

        and : 'The mocked output tensor never returns the mock device because our custom call hook replaces execution.'
            0 * output.getDevice() >> Mock(Device)

        and : 'The ADAnalyzer of the mock implementation will not be called because "doAD" is set to "false".'
            0 * implementation.isSuitableFor(_)

        and : 'The agent creator is never accessed because "doAD" is set to false.'
            0 * implementation.supplyADAgentFor(_,_,_)

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output
    }

    def 'Lambda properties of mock implementation interact with FunctionNode (AbstractFunction) as expected.'()
    {
        given : 'A mock agent.'
            def agent = Mock(ADAgent)

        and : 'A new operation type with a new implementation.'
            def type = Mock(Operation)

        and : 'A list of function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output, an input and a graph node.'
            Tsr output = Mock(Tsr)
            Tsr input = Mock(Tsr)
            Device device = Mock(Device)
            GraphNode node = Mock(GraphNode)
            def ndc = Mock(NDConfiguration)
            var mutate = Mock(AbstractTensor.Unsafe)

        and : 'A mocked operation implementation.'
            def implementation = Mock(Algorithm)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, true)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.getArity() >> 2
            function.isFlat()
            function.isDoingAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call([input])

        then : 'The custom call hook is being accessed as outlined below.'
            (1.._) * input.getNDConf() >> ndc
            (1.._) * ndc.shape() >> new int[]{1,2}
            (1.._) * type.isInline() >> false
            (1.._) * type.getAlgorithmFor(_) >> implementation
            (1.._) * implementation.dispatch(_,_) >> output
            (1.._) * output.getUnsafe() >> mutate
            (1.._) * mutate.setIsIntermediate(false) >> output
            (1.._) * output.isIntermediate() >> true
            (1.._) * device.access( _ ) >> Mock(Device.Access)

        and : 'The GraphNode instance which will be created as tensor component interacts as follows.'
            (1.._) * input.getGraphNode() >> node
            (0.._) * input.get(GraphNode) >> node
            (1.._) * node.getLock() >> Mock(GraphLock)
            (1.._) * input.getDevice() >> device // Device is being queried for execution...
            _ * type.getOperator() >> 'test_identifier'
            (1.._) * output.getDevice() >> device

        and : 'The given ADAnalyzer instance is being called because auto-differentiation is enabled.'
            (1.._) * input.rqsGradient() >> true
            (1.._) * implementation.canPerformForwardADFor(_) >> false
            (1.._) * implementation.canPerformBackwardADFor(_) >> true
            (1.._) * node.getPayload() >> input
            (1.._) * node.usesAD() >> true

        and : 'The agent creator is being accessed because "doAD" is set to true and the input requires gradients.'
            1 * implementation.supplyADAgentFor(_,_,_) >> agent
            1 * agent.derivative() >> null

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output

    }



}
