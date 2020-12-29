package ut.calculus


import neureka.backend.api.operations.AbstractOperation
import neureka.backend.api.operations.OperationContext
import spock.lang.Specification

class OperationContext_Unit_Tests extends Specification
{
    def 'OperationContext instances can be created by cloning from Singleton instance.'() {

        given : 'The singleton OperationContext instance and a OperationType mock.'
            def mockOperation = Mock(AbstractOperation)
            def context = OperationContext.get()

        when : 'A clone is being created by calling "clone()" on the given context...'
            def clone = context.clone()

        then : 'The two instances are not the same objects.'
            clone != context

        and : 'They contain the same entries.'
            clone.id() == context.id()
            clone.lookup() == context.lookup()
            clone.instances() == context.instances()

        when : 'The clone is changes its state.'
            clone.incrementID()
            clone.lookup().put("TEST KEY", null)
            clone.instances().add( mockOperation )

        then : 'Their properties will no longer be the same.'
            clone.id() != context.id()
            clone.lookup() != context.lookup()
            clone.instances() != context.instances()

        and : 'The change will be as expected.'
            clone.id() == context.id() + 1
            clone.lookup().size() == context.lookup().size() + 1
            clone.lookup().containsKey("TEST KEY")
            clone.instances().size() == context.instances().size() + 1

    }



}
