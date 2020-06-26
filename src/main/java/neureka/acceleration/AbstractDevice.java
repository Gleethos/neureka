package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
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
            _execute_recursively( new Tsr[]{tsrs[offset], tsrs[1+offset]}, OperationType.instance("idy"), -1 );
        }
        else if ( type.identifier().equals(">") )
        {
            int offset = ( tsrs[0] == null ) ? 1 : 0;
            _execute_recursively( new Tsr[]{tsrs[1+offset], tsrs[offset]}, OperationType.instance("idy"), -1 );
        }
        else
        {
            _createNewDrainTensorIn(this, tsrs, type);
            if (
                    tsrs.length == 3 && d<0 && // TODO: make a TypeExecutor for this!!!!
                            (
                                    tsrs[1].isVirtual() || tsrs[2].isVirtual() ||
                                    (
                                            !tsrs[1].isOutsourced() && tsrs[1].size() == 1
                                                    ||
                                            !tsrs[2].isOutsourced() && tsrs[2].size() == 1
                                    )
                            )
            ) {
                if (tsrs[2].isVirtual() || tsrs[2].size() == 1) {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[1]}, OperationType.instance("idy"), -1);
                    _enqueue(tsrs[0], tsrs[2].value64()[0], d, type);
                } else {
                    _execute_recursively(new Tsr[]{tsrs[0], tsrs[2]}, OperationType.instance("idy"), -1);
                    _enqueue(tsrs[0], tsrs[1].value64()[0], d, type);
                }
            } else _execute_recursively(tsrs, type, d);
        }
        return this;
    }

    private Tsr _execute_recursively( Tsr[] tsrs, OperationType type, int d )
    {
        OperationTypeImplementation.ExecutionCall call = new OperationTypeImplementation.ExecutionCall(this, tsrs, d, type);
        OperationTypeImplementation<Object> executor = call.getExecutor();
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
            Tsr output = new Tsr(shp, 0.0);
            device.add(output);
            tsrs[0] = output;
        }
    }

}
