package ut.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.autograd.ADAgent
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
import neureka.calculus.environment.ExecutionCall
import neureka.calculus.environment.OperationType
import neureka.calculus.environment.OperationTypeImplementation
import neureka.calculus.environment.implementations.GenericImplementation
import neureka.calculus.environment.implementations.Operator
import neureka.calculus.environment.operations.OperationContext
import neureka.calculus.factory.assembly.FunctionBuilder
import neureka.calculus.factory.assembly.FunctionNode
import neureka.calculus.factory.components.FunctionInput
import org.junit.Ignore
import spock.lang.Specification

class Calculus_Extension_Unit_Tests extends Specification
{

   /*     // TODO
    def 'GEMM matrix multiplication reference implementation can be set as custom OperationType.'(){
    }
    */

    def 'Mock operation interacts with FunctionNode (AbstractFunction) instance as expected.'(){

        given : 'Neureka is being reset.'
            Neureka.instance().reset()

        and : 'A new operation type with a new implementation.'
            def type = Mock(OperationType)

        and : 'A list of function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output'
            Tsr output = Mock(Tsr)

        and : 'A mocked operation implementation.'
            def implementation = Mock(OperationTypeImplementation)

        and : 'A custom call hook for said implementation.'
            def customCallHook = Mock(OperationTypeImplementation.InitialCallHook)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, false)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.arity() >> 2
            function.isFlat()
            !function.doesAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call(new Tsr[0])

        then : 'The custom call hook is being accessed as outlined below.'
            (1.._) * type.implementationOf(_) >> implementation
            (1.._) * implementation.getCallHook() >> customCallHook
            (1.._) * customCallHook.handle(_,_) >> output
        and : 'The GraphNode instance which will be created as tensor component acts as follows.'
            0 * output.device() >> Mock(Device)

        and : 'The ADAnalyzer of the mock implementation will not be called because "doAD" is set to "false".'
            0 * implementation.getADAnalyzer() >> Mock(OperationTypeImplementation.ADAnalyzer)

        and : 'The agent creator is never accessed because "doAD" is set to false.'
            0 * implementation.getADAgentCreator()

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output

    }

    def 'Lambda properties of mock implementation interacts with FunctionNode (AbstractFunction) as expected.'()
    {
        given : 'Neureka is being reset.'
            Neureka.instance().reset()

        and : 'A mock agent.'
            def agentSupplier = Mock(OperationTypeImplementation.ADAgentSupplier)
            def agent = Mock(ADAgent)

        and : 'A new operation type with a new implementation.'
            def type = Mock(OperationType)

        and : 'A list of function source nodes.'
            def children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output'
            Tsr output = Mock(Tsr)
            Tsr input = Mock(Tsr)
            GraphNode node = Mock(GraphNode)

        and : 'A mocked operation implementation.'
            def implementation = Mock(OperationTypeImplementation)

        and : 'A custom call hook for said implementation.'
            def customCallHook = Mock(OperationTypeImplementation.InitialCallHook)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            def function = new FunctionNode(type, children, true)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * type.arity() >> 2
            function.isFlat()
            function.doesAD()

        when : 'The function is being called with an empty tensor array...'
            def result = function.call([input])

        then : 'The custom call hook is being accessed as outlined below.'
            (1.._) * type.implementationOf(_) >> implementation
            (1.._) * implementation.getCallHook() >> customCallHook
            (1.._) * customCallHook.handle(_,_) >> output
        and : 'The GraphNode instance which will be created as tensor component acts as follows.'
            (1.._) * input.find(GraphNode.class) >> node
            (1.._) * node.lock() >> Mock(GraphLock)
            (1.._) * input.device() >> Mock(Device)
            _ * type.identifier() >> 'test_identifier'
            (1.._) * output.device() >> Mock(Device)

        and : 'The given ADAnalyzer instance is being called.'
            (1.._) * input.rqsGradient() >> true
            (1.._) * implementation.getADAnalyzer() >> Mock(OperationTypeImplementation.ADAnalyzer)
            (1.._) * node.getPayload() >> input
            (1.._) * node.usesAD() >> true

        and : 'The agent creator is accessed because "doAD" is set to true and the input requires gradients.'
            1 * implementation.getADAgentCreator() >> agentSupplier
            1 * agentSupplier.getADAgentOf(_,_,_) >> agent
            1 * agent.derivative() >> null

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output

    }


    //@Ignore
    //def 'WIP - Custom call implementation is being called.'(){
//
    //    given : 'Neureka is being reset.'
    //        Neureka.instance().reset()
//
    //    and : 'A new OperationContext for testing.'
    //        OperationContext oldContext = OperationContext.instance()
    //        OperationContext context = OperationContext.instance().clone()
    //        OperationContext.setInstance(context)
//
    //    and : 'A mock agent.'
    //        ADAgent agent = Mock()
//
    //    and : 'A new operation type with a new implementation.'
    //        def type = new OperationType(
    //                "test_operation", "o", 2,
    //                false, false, false, false, false
    //        ).setImplementation(
    //                GenericImplementation.class,
    //                new GenericImplementation()
    //                        .setHandleChecker(call -> true)
    //                        .setADAnalyzer(call -> false)
    //                        .setADAgentCreator(
    //                                (Function f, ExecutionCall<Device> call, boolean forward ) ->  agent
    //                        ).setCallHock(
    //                            ( caller, call ) -> { return null; }
    //                        ).setRJAgent(
    //                                ( call, goDeeperWith ) -> { return null; }
    //                        ).setDrainInstantiation(
    //                                call -> {return call;}
    //                        )
    //        )
//
    //    and : 'A mock FunctionNode '
    //        def children = [Mock(FunctionInput), Mock(FunctionInput)]
    //        def function = new FunctionNode(type, children, false)
//
//
    //    when : ''
    //    function.call(new Tsr[0])
//
    //    then : ''
    //    assert true
//
//
    //    cleanup :
    //    OperationContext.setInstance(oldContext)
//
    //}



}
