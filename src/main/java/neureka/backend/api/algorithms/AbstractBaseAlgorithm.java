package neureka.backend.api.algorithms;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.implementations.ImplementationFor;
import neureka.backend.api.operations.Operation;
import neureka.devices.Device;
import neureka.backend.api.ExecutionCall;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

@Accessors( prefix = {"_"} )
public abstract class AbstractBaseAlgorithm<FinalType> implements Algorithm<FinalType>
{
    @Getter private final String _name;

    protected final Map< Class< Device<?> >, ImplementationFor< Device<?> >> _executions = new HashMap<>();

    public AbstractBaseAlgorithm(String name) { _name = name; }

    @Override
    public Tsr recursiveReductionOf(
            ExecutionCall<Device> call,
            Consumer<ExecutionCall<Device>> finalExecution
    ) {
        Device device = call.getDevice();
        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        Operation type = call.getOperation();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[ i ] != null && !tsrs[ i ].isOutsourced() ) {
                try {
                    device.store(tsrs[i]);
                } catch ( Exception e ) {
                    e.printStackTrace();
                }

                rollbacks[ i ] = tensor -> {
                    try {
                    device.restore( tensor );
                    } catch ( Exception e ) {
                        e.printStackTrace();
                    }
                };

            }
            else rollbacks[ i ] = t -> {};
        }
        /* For the following operations with the correct arity RJAgent should do: ...
            case ("s" + ((char) 187)): tsrs = new Tsr[]{tsrs[ 2 ], tsrs[ 1 ], tsrs[ 0 ]};
            case ("d" + ((char) 187)): tsrs = new Tsr[]{tsrs[ 2 ], tsrs[ 1 ], tsrs[ 0 ]};
            case ("p" + ((char) 187)): tsrs = new Tsr[]{tsrs[ 2 ], tsrs[ 1 ], tsrs[ 0 ]};
            case ("m" + ((char) 187)): tsrs = new Tsr[]{tsrs[ 2 ], tsrs[ 1 ], tsrs[ 0 ]};
            case ">": tsrs = new Tsr[]{tsrs[ 1 ], tsrs[ 0 ]};
         */
        /*
            Below is the core lambda of recursive preprocessing
            which is defined for each OperationImplementation individually :
         */
        Tsr result = handleRecursivelyAccordingToArity( call, c -> recursiveReductionOf( c, finalExecution ) );
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
    public <D extends Device<?>, E extends ImplementationFor<D>> FinalType setImplementationFor( Class<D> deviceClass, E implementation ) {
        _executions.put(
                (Class<Device<?>>) deviceClass,
                (ImplementationFor<Device<?>>) implementation
        );
        return (FinalType) this;
    }

    @Override
    public <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass ) {
        return (ImplementationFor<D>) _executions.get( deviceClass );
    }

}
