package neureka.calculus.environment.implementations;

import neureka.acceleration.Device;
import neureka.calculus.environment.ExecutionCall;

public class GenericImplementation extends AbstractOperationTypeImplementation< GenericImplementation >
{

    private HandleChecker _canHandle;

    public GenericImplementation(
            HandleChecker canHandle,
            InitialCallHook hook,
            ADAnalyzer analyzer,
            RecursiveJunctionAgent RJAgent,
            DrainInstantiation instantiation
    ) {
        super(analyzer, hook, RJAgent, instantiation);
        _canHandle = canHandle;
    }

    @Override
    public boolean canHandle(ExecutionCall<Device> call) {
        return _canHandle.canHandle(call);
    }

}
