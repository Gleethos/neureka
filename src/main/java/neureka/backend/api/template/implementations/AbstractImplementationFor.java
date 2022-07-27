package neureka.backend.api.template.implementations;

import neureka.Tsr;
import neureka.backend.FunImplementationFor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.devices.Device;

public class AbstractImplementationFor< D extends Device<?> > implements FunImplementationFor<D>
{
    private final int _arity;
    private final FunImplementationFor<D> _lambda;

    public AbstractImplementationFor( FunImplementationFor<D> implementationLambda, int arity ) {
        _lambda = implementationLambda;
        _arity = arity;
    }

    @Override
    public Tsr<?> runAndGetFirstTensor(ExecutionCall<D> call ) {
        if ( _arity >= 0 ) {
            int arity1 = call.arity();
            int arity2 = arity1 - 1; // The first tensor might be the output!
            if (arity1 != _arity && arity2 != _arity)
                throw new IllegalArgumentException(
                    "Expected arity "+_arity+" or "+(_arity + 1)+", but encountered arity "+arity1+" for execution call '"+call+"'."
                );
        }
        //assert call.size() == _arity ;
        return _lambda.runAndGetFirstTensor( call );
    }

}
