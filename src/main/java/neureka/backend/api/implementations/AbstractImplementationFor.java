package neureka.backend.api.implementations;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

public class AbstractImplementationFor< TargetDevice extends Device<?> > implements ImplementationFor<TargetDevice>
{
    private int _arity = -1;
    private final ImplementationFor<TargetDevice> _lambda;

    public AbstractImplementationFor( ImplementationFor<TargetDevice> implementationLambda, int arity ) {
        _lambda = implementationLambda;
        _arity = arity;
    }

    @Override
    public void run( ExecutionCall<TargetDevice> call ) {
        //if (call.size() != _arity) System.out.println(call.getOperation().getFunction()+ call.getImplementation().getName()+_arity+"-"+call.size());
        //assert call.size() == _arity ;
        _lambda.run( call );
    }

}
