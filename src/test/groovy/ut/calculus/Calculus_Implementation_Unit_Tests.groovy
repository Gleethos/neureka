package ut.calculus

import neureka.Neureka
import neureka.autograd.ADAgent
import neureka.calculus.Function
import neureka.calculus.environment.ExecutionCall
import neureka.calculus.environment.OperationTypeImplementation
import neureka.calculus.environment.implementations.Operator
import neureka.calculus.environment.operations.OperationContext
import spock.lang.Specification

class Calculus_Implementation_Unit_Tests extends Specification
{
    def 'Operator implementations behave as expected.'(
            OperationTypeImplementation imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'A mock Function.'
            def function = Mock(Function)

        and : 'A mock ExecutionCall.'
            def call = Mock(ExecutionCall)

        when :
            ADAgent agent = imp.ADAgentCreator.getADAgentOf(
                    function,
                    null,
                    call,
                    true
            )

        then :
            agent.isForward()

        when :
            agent = imp.ADAgentCreator.getADAgentOf(
                    function,
                    null,
                    call,
                    false
            )

        then :
            !agent.isForward()

        where : 'The variable "imp" is from a List of OperationType implementations of operator types.'
            imp << OperationContext.instance()
                    .instances()
                    .stream()
                    .filter(
                            e ->
                                    e.isOperator() &&
                                            e.identifier().length()==1 &&
                                                e.supports( Operator.class )
                    ).map( e -> e.getImplementation( Operator.class ) )


    }



}
