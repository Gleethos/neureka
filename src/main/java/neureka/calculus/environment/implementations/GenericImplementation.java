package neureka.calculus.environment.implementations;

import neureka.acceleration.Device;
import neureka.calculus.environment.ExecutionCall;

public class GenericImplementation extends AbstractOperationTypeImplementation< GenericImplementation >
{

    private HandleChecker _canHandle;

    public GenericImplementation(
            HandleChecker canHandle,
            ADAnalyzer analyzer,
            RecursiveJunctionAgent RJAgent,
            DrainInstantiation instantiation
    ) {
        super(analyzer, RJAgent, instantiation);
        _canHandle = canHandle;
    }

    @Override
    public boolean canHandle(ExecutionCall<Device> call) {
        return _canHandle.canHandle(call);
    }

}
