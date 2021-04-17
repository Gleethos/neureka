package ut.backend

import neureka.Neureka
import neureka.Tsr
import neureka.autograd.ADAgent
import neureka.calculus.Function
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Algorithm
import neureka.backend.standard.algorithms.Activation
import neureka.backend.standard.algorithms.Broadcast
import neureka.backend.standard.algorithms.Convolution
import neureka.backend.standard.algorithms.Operator
import neureka.backend.api.operations.OperationContext
import spock.lang.Specification

class Backend_Algorithm_AD_Unit_Tests extends Specification
{

    def 'Operator implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A mock Function.'
            def function = Mock(Function)
            def derivative = Mock(Tsr)
            function.derive(*_) >> derivative

        and : 'A mock ExecutionCall.'
            def call = Mock(ExecutionCall)

        when : 'A new ADAgent is being instantiated by calling the given implementation with these arguments...'
            ADAgent agent = imp.ADAgentSupplier.getADAgentOf(
                    function,
                    call,
                    true
            )

        then : 'The agent is configured to perform forward-AD and it contains the derivative generated by the function!'
            agent.hasForward()
            agent.derivative() == derivative

        when : 'The agent generator is called once more with the forward flag set to false...'
            agent = imp.ADAgentSupplier.getADAgentOf(
                    function,
                    call,
                    false
            )

        then : 'The result is similar except the agent is not configured to perform forward-AD as was the case previously.'
            agent.hasForward()
            agent.derivative() == derivative

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << OperationContext.get()
                    .instances()
                    .stream()
                    .filter(
                            e ->
                                    e.isOperator() &&
                                            e.getOperator().length()==1 &&
                                                e.supports( Operator.class )
                    ).map( e -> e.getAlgorithm( Operator.class ) )
    }


    def 'Activation implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A mock Function.'
            def function = Mock(Function)
            def derivative = Mock(Tsr)
            function.derive(*_) >> derivative

        and : 'A mock ExecutionCall.'
            def call = Mock(ExecutionCall)

        when : 'A new ADAgent is being instantiated by calling the given implementation with these arguments...'
            ADAgent agent = imp.ADAgentSupplier.getADAgentOf(
                function,
                call,
                true
            )

        then : 'The agent is configured to perform forward-AD and it contains the derivative generated by the function!'
            agent.hasForward()
            agent.derivative() == derivative

        when : 'The agent generator is called once more with the forward flag set to false...'
            agent = imp.ADAgentSupplier.getADAgentOf(
                function,
                call,
                false
            )

        then : 'The result is similar except the agent is not configured to perform forward-AD as was the case previously.'
            //!agent.isForward() //TODO: Fix this!
            agent.derivative() == derivative

        where : 'The variable "imp" is from a List of OperationType implementations of type "Activation".'
            imp << OperationContext.get()
                .instances()
                .stream()
                .filter(
                        e ->
                                        e.supports( Activation.class )
                ).map( e -> e.getAlgorithm( Activation.class ) )
    }


    def 'Convolution implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A mock Function.'
            def function = Mock(Function)
            def derivative = Mock(Tsr)
            function.derive(*_) >> derivative

        and : 'A mock ExecutionCall.'
            def call = Mock(ExecutionCall)

        when : 'A new ADAgent is being instantiated by calling the given implementation with these arguments...'
            ADAgent agent = imp.ADAgentSupplier.getADAgentOf(
                    function,
                    call,
                    true
            )

        then : 'An exception is being thrown because implementations of type "Convolution" can only perform reverse mode AD!'
            def exception = thrown(IllegalArgumentException)
            exception.message == "Convolution of does not support forward-AD!"

        when : 'The agent generator is called once more with the forward flag set to false...'
            agent = imp.ADAgentSupplier.getADAgentOf(
                function,
                call,
                false
            )

        then : 'No exception is being thrown and the agent is configured to perform backward-AD.'
            //!agent.isForward() //TODO: Fix tis
            agent.derivative() == derivative

        where : 'The variable "imp" is from a List of OperationType implementations of type "Convolution".'
            imp << OperationContext.get()
                .instances()
                .stream()
                .filter(
                        e ->
                                e.isOperator() &&
                                        e.getOperator().length()==1 &&
                                        e.supports( Convolution.class )
                ).map( e -> e.getAlgorithm( Convolution.class ) )
    }



    def 'Broadcast implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A mock Function.'
            def function = Mock( Function )
            def derivative = Mock( Tsr )
            function.derive(*_) >> derivative

        and : 'A mock ExecutionCall.'
            def call = Mock( ExecutionCall )

        when : 'A new ADAgent is being instantiated by calling the given implementation with these arguments...'
            ADAgent agent = imp.ADAgentSupplier.getADAgentOf(
                function,
                call,
                true
            )

        then : 'An exception is being thrown because implementations of type "Broadcast" can only perform reverse mode AD!'
            def exception = thrown( IllegalArgumentException )
            exception.message == "Broadcast implementation does not support forward-AD!"

        when : 'The agent generator is called once more with the forward flag set to false...'
            agent = imp.ADAgentSupplier.getADAgentOf(
                function,
                call,
                false
            )

        then : 'No exception is being thrown and the agent is configured to perform backward-AD.'
            agent.hasForward()
            agent.derivative() == derivative

        where : 'The variable "imp" is from a List of OperationType implementations of type "Convolution".'
            imp << OperationContext.get()
                .instances()
                .stream()
                .filter(
                        e ->
                                e.isOperator() &&
                                        e.supports( Broadcast.class )
                ).map( e -> e.getAlgorithm( Broadcast.class ) )
    }



}
