package neureka.calculus.environment.implementations;

public class GenericImplementation extends AbstractOperationTypeImplementation< GenericImplementation >
{

    public GenericImplementation(
            HandleChecker canHandle,
            InitialCallHook hook,
            ADAnalyzer analyzer,
            RecursiveJunctionAgent RJAgent,
            DrainInstantiation instantiation
    ) {
        super(analyzer, hook, RJAgent, instantiation);
        setHandleChecker(canHandle);
    }

}
