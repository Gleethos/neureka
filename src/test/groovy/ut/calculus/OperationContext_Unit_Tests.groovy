package ut.calculus

import neureka.calculus.backend.operations.AbstractOperationType
import neureka.calculus.backend.operations.OperationContext
import spock.lang.Specification

class OperationContext_Unit_Tests extends Specification
{
    def 'OperationContext instances can be created by cloning from Singleton instance.'() {

        given : 'The singleton OperationContext instance and a OperationType mock.'
            def mockOperation = Mock(AbstractOperationType)
            def context = OperationContext.instance()

        when : 'A clone is being created by calling "clone()" on the given context...'
            def clone = context.clone()

        then : 'The two instances are not the same objects.'
            clone != context

        and : 'They contain the same entries.'
            clone.getID() == context.getID()
            clone.getLookup() == context.getLookup()
            clone.getRegister() == context.getRegister()

        when : 'The clone is changes its state.'
            clone.incrementID()
            clone.getLookup().put("TEST KEY", null)
            clone.getRegister().add( mockOperation )

        then : 'Their properties will no longer be the same.'
            clone.getID() != context.getID()
            clone.getLookup() != context.getLookup()
            clone.getRegister() != context.getRegister()

        and : 'The change will be as expected.'
            clone.getID() == context.getID() + 1
            clone.getLookup().size() == context.getLookup().size() + 1
            clone.getLookup().containsKey("TEST KEY")
            clone.getRegister().size() == context.getRegister().size() + 1

    }



}
