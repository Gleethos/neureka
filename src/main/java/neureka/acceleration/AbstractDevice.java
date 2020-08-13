package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;

import java.lang.ref.Cleaner;

public abstract class AbstractDevice implements Device, Component<Tsr>
{
    private static final Cleaner _CLEANER = Cleaner.create();

    protected abstract void _enqueue(Tsr[] tsrs, int d, OperationType type);

    protected abstract void _enqueue(Tsr t, double value, int d, OperationType type);

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
        call = call.getImplementation().getDrainInstantiation().handle(call);
        OperationTypeImplementation<Object> executor = call.getImplementation();
        executor.recursiveReductionOf(
                call,
                c -> _enqueue(c.getTensors(), c.getDerivativeIndex(), c.getType())
        );
        return this;
    }

}
