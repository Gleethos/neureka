package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;

import java.lang.ref.Cleaner;

public abstract class AbstractDevice implements  Device, Component<Tsr>
{
    private static final Cleaner CLEANER = Cleaner.create();

    protected abstract void _enqueue(Tsr[] tsrs, int d, OperationType type);

    protected abstract void _enqueue(Tsr t, double value, int d, OperationType type);

    @Override
    public void update(Tsr oldOwner, Tsr newOwner){
        swap(oldOwner, newOwner);
    }

    @Override
    public Device cleaning(Tsr tensor, Runnable action){
        CLEANER.register(tensor, action);
        return this;
    }

    protected void _cleaning(Object o, Runnable action){
        CLEANER.register(o, action);
    }

    @Override
    public Device execute(Tsr[] tsrs, OperationType type, int d)
    {
        if ( type.identifier().equals("<") )
        {
            int offset = ( tsrs[0] == null ) ? 1 : 0;
            _execute( new Tsr[]{tsrs[offset], tsrs[1+offset]}, OperationType.instance("idy"), -1 );
        }
        else
        if ( type.identifier().equals(">") )
        {
            int offset = ( tsrs[0] == null ) ? 1 : 0;
            _execute( new Tsr[]{tsrs[1+offset], tsrs[offset]}, OperationType.instance("idy"), -1 );
        }
        else
        {
            _createNewDrainTensorIn(this, tsrs, type);
            _execute(tsrs, type, d);
        }
        return this;
    }

    private Tsr _execute(Tsr[] tsrs, OperationType type, int d )
    {
        ExecutionCall call = new ExecutionCall(this, tsrs, d, type);
        OperationTypeImplementation<Object> executor = call.getImplementation();
        executor.reduce (
                call,
                c -> _enqueue(c.getTensors(), c.getDerivativeIndex(), c.getType())
        );
        return call.getTensors()[0];
    }


    private static void _createNewDrainTensorIn(Device device, Tsr[] tsrs, OperationType type)
    {
        if ( tsrs[0] == null )// Creating a new tensor:
        {
            int[] shp = (type.identifier().endsWith("x"))
                    ? Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape())
                    : tsrs[1].getNDConf().shape();
            Tsr output = new Tsr( shp, 0.0 );
            output.setIsVirtual(false); // This tensor will be 'filled'! Therefore : needs to be 'whole!
            device.add(output);
            tsrs[0] = output;
        }
    }

}
