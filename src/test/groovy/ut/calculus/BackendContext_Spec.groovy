package ut.calculus

import neureka.Neureka
import neureka.backend.api.Operation
import neureka.backend.api.BackendContext
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

import java.util.function.Function

@Title("The BackendContext is a cloneable context which can run Tasks.")
@Narrative('''
    
    This specification defines the expected behaviour of the backend context
    which should expose a convenient API to work with.
    This API should allow for tasks to be running on a given context
    which is important for testing and modularity not only
    during library startup but also throughout the runtime.
    
''')
class BackendContext_Spec extends Specification
{
    def 'BackendContext instances can be created by cloning from Singleton instance.'()
    {
        given : 'The singleton BackendContext instance and a OperationType mock.'
            def mockOperation = Mock(Operation)
            def context = Neureka.get().backend()

        when : 'A clone is being created by calling "clone()" on the given context...'
            def clone = context.clone()

        then : 'The two instances are not the same objects.'
            clone != context

        and : 'They contain the same entries.'
            clone.size()      == context.size()
            clone.getOperationLookupMap()    == context.getOperationLookupMap()
            clone.getOperations() == context.getOperations()

        when :
            1 * mockOperation.getOperator() >> ""
            1 * mockOperation.getIdentifier() >> ""

        and : 'The clone is changes its state.'
            clone.addOperation( mockOperation )

        then : 'Their properties will no longer be the same.'
            clone.size() != context.size()
            clone.getOperationLookupMap() != context.getOperationLookupMap()
            clone.getOperations() != context.getOperations()

        and : 'The change will be as expected.'
            clone.size() == context.size() + 1
            clone.getOperationLookupMap().size() == context.getOperationLookupMap().size() + 1
            clone.getOperationLookupMap().containsKey("")
            clone.getOperations().size() == context.getOperations().size() + 1

    }


    def 'BackendContext instances return Runner instances for easy visiting.'()
    {
        given : 'The current thread local BackendContext instance.'
            def current = Neureka.get().backend()

        and : 'A clone is being created by calling "clone()" on the given context...'
            def clone = current.clone()

        and: 'We ger a Runner instance via the following method:'
            def run = clone.runner()

        and : 'Also, we create a mocked lambda function as a "spy"!'
            def spy = Mock(Function)

        when : 'We pass a lambda to the "useFor" method to the runner, containing a closure with the spy...'
            run.run({spy.apply(Neureka.get().backend())})

        then : 'The spy will tell us that the passed lambda has been executed by the runner in the clone context!'
            1 * spy.apply(clone)

        and : 'The context accessible through the static "get" method will indeed be the current context!'
            current == Neureka.get().backend()

    }


    def 'BackendContext instances return Runner instances for easy visiting with return values.'(
            Closure<Object> runWrapper
    ) {

        given : 'The current thread local BackendContext instance.'
            def current = Neureka.get().backend()

        and : 'A clone is being created by calling "clone()" on the given context...'
            def clone = current.clone()

        and: 'We wrap a Runner instance around a wrapper which will test its methods!'
            def run = runWrapper( clone.runner() )

        when : 'Querying the thread local context inside the Runner...'
            def innerContext = run { Neureka.get().backend() }
        and : '...also outside the Runner lambda...'
            def outerContext = Neureka.get().backend()

        then : 'These two context instances will be different objects!'
            innerContext != outerContext

        and : 'The inner context will in fact be the clone which provided the Runner!'
            innerContext == clone

        and : 'The outer context is as expected simply the current context.'
            outerContext == current

        where : 'The following conceptually identical Runner methods can be used:'
            runWrapper << [
                    (BackendContext.Runner runner) -> { (arg) -> runner.call(arg) },
                    (BackendContext.Runner runner) -> { (arg) -> runner.invoke(arg) },
                    (BackendContext.Runner runner) -> { (arg) -> runner.runAndGet(arg) }
            ]
    }


}
