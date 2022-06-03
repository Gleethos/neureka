package neureka.backend.api.template.implementations;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

public class AbstractImplementationFor< TargetDevice extends Device<?> > implements ImplementationFor<TargetDevice>
{
    private final int _arity;
    private final ImplementationFor<TargetDevice> _lambda;

    public AbstractImplementationFor( ImplementationFor<TargetDevice> implementationLambda, int arity ) {
        _lambda = implementationLambda;
        _arity = arity;
    }

    @Override
    public void run( ExecutionCall<TargetDevice> call ) {
        if ( _arity >= 0 ) {
            int arity1 = call.arity();
            int arity2 = arity1 - 1; // The first tensor might be the output!
            if (arity1 != _arity && arity2 != _arity)
                throw new IllegalArgumentException(
                    "Expected arity "+_arity+" or "+(_arity + 1)+", but encountered arity "+arity1+" for execution call '"+call+"'."
                );
        }
        //assert call.size() == _arity ;
        _lambda.run( call );
    }

}
