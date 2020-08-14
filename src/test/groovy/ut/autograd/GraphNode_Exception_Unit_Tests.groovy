package ut.autograd

import neureka.Tsr
import neureka.autograd.GraphLock
import neureka.autograd.GraphNode
import neureka.calculus.Function
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



}
