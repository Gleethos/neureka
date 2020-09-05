package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.OperationTypeImplementation;

import java.lang.ref.Cleaner;

public abstract class AbstractDevice implements Device, Component<Tsr>
{
    private static final Cleaner _CLEANER = Cleaner.create();

    protected abstract void _enqueue(Tsr[] tsrs, int d, OperationType type);

    @Override
    public void update(Tsr oldOwner, Tsr newOwner){
        swap(oldOwner, newOwner);
    }

    @Override
    public Device cleaning(Tsr tensor, Runnable action){
        _CLEANER.register(tensor, action);
        return this;
    }

    protected void _cleaning(Object o, Runnable action){
        _CLEANER.register(o, action);
    }

    @Override
    public Device execute( ExecutionCall call )
    {
        call = call.getImplementation().instantiateNewTensorsForExecutionIn(call);
        for ( Tsr t : call.getTensors() ) {
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );
        }
        ((OperationTypeImplementation<Object>)call.getImplementation())
                .recursiveReductionOf(
                    call,
                    c -> _enqueue(c.getTensors(), c.getDerivativeIndex(), c.getType())
                );
        return this;
    }

}
