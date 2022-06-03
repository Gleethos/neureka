package ut.backend

import neureka.backend.api.Algorithm
import neureka.backend.api.template.algorithms.AbstractFunctionalAlgorithm
import neureka.backend.api.template.algorithms.fun.AutoDiffMode
import neureka.backend.api.template.algorithms.fun.Execution
import neureka.backend.api.template.algorithms.fun.ExecutionPreparation
import neureka.backend.api.template.algorithms.fun.SuitabilityPredicate
import spock.lang.Specification

import java.util.function.Consumer

class Backend_Functional_Algorithm_Spec extends Specification
{

    def 'A functional algorithm cannot be used if it was not built properly!'(
            Consumer<Algorithm> caller
    ) {

        given : 'We create a dumb test algorithm.'
            def algorithm = new TestAlgorithm()

        when : 'We call a method on the algorithm...'
            caller(algorithm)

        then : 'This should throw an illegal state exception, simply because we have not built the algorithm properly!'
            def exception = thrown(IllegalStateException)
        and : 'The exception tells us this:'
            exception.message == "Trying use an instance of 'TestAlgorithm' with name 'test_name' which was not fully built!"

        where : 'We call the following methods:'
            caller << [
                    { Algorithm it -> it.autoDiffModeFrom(null) },
                    { Algorithm it -> it.execute(null, null) },
                    { Algorithm it -> it.prepare(null) },
                    { Algorithm it -> it.supplyADAgentFor(null, null) }
            ]

    }


    def 'A functional algorithm does not accept null as an answer!'() {

        given : 'We create a dumb test algorithm.'
            def algorithm = new TestAlgorithm()

        when : 'We build it thoroughly...'
            algorithm
                    .setIsSuitableFor(call -> SuitabilityPredicate.EXCELLENT)
                    .setAutogradModeFor( call ->  AutoDiffMode.BACKWARD_ONLY )
                    .setExecution(( caller, call ) -> null)
                    .setCallPreparation(call -> null)
                    .buildFunAlgorithm()

        then : 'The algorithm should be usable just fine!'
            algorithm.isSuitableFor(null) == SuitabilityPredicate.EXCELLENT
            algorithm.autoDiffModeFrom(null) == AutoDiffMode.BACKWARD_ONLY
            algorithm.execute(null, null) == null
            algorithm.prepare(null) == null

        when : 'We create a new instance!'
            algorithm = new TestAlgorithm()
        and : 'Which we do not build fully this time...'
            algorithm
                    .setIsSuitableFor(call -> SuitabilityPredicate.EXCELLENT)
                    .setAutogradModeFor( call ->  AutoDiffMode.BACKWARD_ONLY )
                    .setExecution(( caller, call ) -> null)
                    .setCallPreparation(null) // This is not acceptable!
                    .buildFunAlgorithm()

        then : 'This should throw an illegal state exception, because we have not built the algorithm properly!'
            def exception = thrown(IllegalStateException)
        and : 'The exception tells us this:'
            exception.message == "Instance 'TestAlgorithm' incomplete!"
    }


    def 'A functional algorithm warns us when modified after it has been built!'(
            Class<?> type, Consumer<TestAlgorithm> setter
    ) {
        given : 'We create a dumb test algorithm.'
            def algorithm = new TestAlgorithm()
        and :
            def oldStream = System.err
            System.err = Mock(PrintStream)

        when : 'We build it thoroughly...'
            algorithm
                    .setIsSuitableFor(call -> SuitabilityPredicate.EXCELLENT)
                    .setAutogradModeFor( call ->  AutoDiffMode.BACKWARD_ONLY )
                    .setExecution(( caller, call ) -> null)
                    .setCallPreparation(call -> null)
                    .buildFunAlgorithm()

        then : 'The algorithm should be usable just fine!'
            algorithm.isSuitableFor(null) == SuitabilityPredicate.EXCELLENT
            algorithm.autoDiffModeFrom(null) == AutoDiffMode.BACKWARD_ONLY
            algorithm.execute(null, null) == null
            algorithm.prepare(null) == null

        when : 'We try to modify the algorithm even if it is already built...'
            setter(algorithm)
        then : 'We will get a warning which wells us that mutating the state of the algorithm is discouraged!'
            1 * System.err.println(
                    "[Test worker] WARN neureka.backend.api.template.algorithms.AbstractFunctionalAlgorithm - " +
                    "Implementation '$type.simpleName' in algorithm '$algorithm' was modified! " +
                    "Please consider only modifying the standard backend state of Neureka for experimental reasons."
                )

        cleanup :
            System.err = oldStream

        where :
            type                        | setter
            ExecutionPreparation.class  | { TestAlgorithm it -> it.setCallPreparation( call -> null ) }
            SuitabilityPredicate.class  | { TestAlgorithm it -> it.setIsSuitableFor( call -> SuitabilityPredicate.NOT_GOOD ) }
            Execution.class             | { TestAlgorithm it -> it.setExecution(( caller, call ) -> null ) }
    }


    class TestAlgorithm extends AbstractFunctionalAlgorithm<TestAlgorithm> {
        TestAlgorithm() {
            super("test_name")
        }
    }

}
