package neureka.calculus.backend.implementations;

import neureka.Tsr;
import neureka.device.Device;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.executions.ExecutorFor;
import neureka.calculus.backend.operations.OperationType;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractBaseOperationTypeImplementation<FinalType> implements OperationTypeImplementation<FinalType>
{
    private final String _name;

    protected final Map< Class<ExecutorFor< Device >>, ExecutorFor< Device > > _executions = new HashMap<>();

    public AbstractBaseOperationTypeImplementation(String name) { _name = name; }


    @Override
    public String getName(){
        return _name;
    }


    @Override
    public Tsr recursiveReductionOf(
            ExecutionCall<Device> call,
            Consumer<ExecutionCall<Device>> finalExecution
    ) {
        Device device = call.getDevice();
        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        OperationType type = call.getType();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[ i ] != null && !tsrs[ i ].isOutsourced() ) {
                device.add(tsrs[ i ]);
                rollbacks[ i ] = device::get;
            }
            else rollbacks[ i ] = t -> {};
        }

        /* For the following operations with the correct arity RJAgent should do: ...
            case ("s" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[ 0 ]};
            case ("d" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[ 0 ]};
            case ("p" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[ 0 ]};
            case ("m" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[ 0 ]};
            case ">": tsrs = new Tsr[]{tsrs[1], tsrs[ 0 ]};
         */
        /*
            Below is the core lambda of recursive preprocessing
            which is defined for each OperationImplementation individually :
         */
        Tsr result = handleRecursivelyAccordingToArity(call, c -> recursiveReductionOf( c, finalExecution ));
        if ( result == null ) {
            finalExecution.accept(
                    new ExecutionCall<>( device, call.getTensors(), d, type )
            );
        }
        else return result;


        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[ i ] != null && !tsrs[ i ].isUndefined() ) rollbacks[ i ].accept(tsrs[ i ]);
        }
        return tsrs[ 0 ];
    }


    //---

    @Override
    public <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution){
        _executions.put(
                (Class<ExecutorFor<Device>>) deviceClass,
                (ExecutorFor<Device>) execution
        );
        return (FinalType) this;
    }

    @Override
    public <D extends Device, E extends ExecutorFor<D>> E getExecutor(Class<E> deviceClass){
        return (E) _executions.get(deviceClass);
    }

}
