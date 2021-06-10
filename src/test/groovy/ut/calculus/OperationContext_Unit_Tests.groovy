package ut.calculus

import neureka.Neureka
import neureka.backend.api.operations.AbstractOperation
import neureka.backend.api.operations.OperationContext
import spock.lang.Specification

import java.util.function.Function

class OperationContext_Unit_Tests extends Specification
{
    def 'OperationContext instances can be created by cloning from Singleton instance.'()
    {
        given : 'The singleton OperationContext instance and a OperationType mock.'
            def mockOperation = Mock(AbstractOperation)
            def context = Neureka.get().context()

        when : 'A clone is being created by calling "clone()" on the given context...'
            def clone = context.clone()

        then : 'The two instances are not the same objects.'
            clone != context

        and : 'They contain the same entries.'
            clone.size() == context.size()
            clone.lookup() == context.lookup()
            clone.instances() == context.instances()

        when : 'The clone is changes its state.'
            clone.incrementID()
            clone.lookup().put("TEST KEY", null)
            clone.instances().add( mockOperation )

        then : 'Their properties will no longer be the same.'
            clone.size() != context.size()
            clone.lookup() != context.lookup()
            clone.instances() != context.instances()

        and : 'The change will be as expected.'
            clone.size() == context.size() + 1
            clone.lookup().size() == context.lookup().size() + 1
            clone.lookup().containsKey("TEST KEY")
            clone.instances().size() == context.instances().size() + 1

    }


    def 'OperationContext instances return Runner instances for easy visiting.'()
    {
        given : 'The current thread local OperationContext instance.'
            def current = Neureka.get().context()

        and : 'A clone is being created by calling "clone()" on the given context...'
            def clone = current.clone()

        and: 'We ger a Runner instance via the following method:'
            def run = clone.runner()

        and : 'Also, we create a mocked lambda function as a "spy"!'
            def spy = Mock(Function)

        when : 'We pass a lambda to the "useFor" method to the runner, containing a closure with the spy...'
            run.run({spy.apply(Neureka.get().context())})

        then : 'The spy will tell us that the passed lambda has been executed by the runner in the clone context!'
            1 * spy.apply(clone)

        and : 'The context accessible through the static "get" method will indeed be the current context!'
            current == Neureka.get().context()

    }


    def 'OperationContext instances return Runner instances for easy visiting with return values.'(
            Closure<Object> runWrapper
    ) {

        given : 'The current thread local OperationContext instance.'
            def current = Neureka.get().context()

        and : 'A clone is being created by calling "clone()" on the given context...'
            def clone = current.clone()

        and: 'We wrap a Runner instance around a wrapper which will test its methods!'
            def run = runWrapper( clone.runner() )

        when : 'Querying the thread local context inside the Runner...'
            def innerContext = run { Neureka.get().context() }
        and : '...also outside the Runner lambda...'
            def outerContext = Neureka.get().context()

        then : 'These two context instances will be different objects!'
            innerContext != outerContext

        and : 'The inner context will in fact be the clone which provided the Runner!'
            innerContext == clone

        and : 'The outer context is as expected simply the current context.'
            outerContext == current

        where : 'The following conceptually identical Runner methods can be used:'
            runWrapper << [
                    (OperationContext.Runner runner) -> { (arg) -> runner.call(arg) },
                    (OperationContext.Runner runner) -> { (arg) -> runner.invoke(arg) },
                    (OperationContext.Runner runner) -> { (arg) -> runner.runAndGet(arg) }
            ]
    }


}
