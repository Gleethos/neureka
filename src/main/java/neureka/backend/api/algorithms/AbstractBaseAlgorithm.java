package neureka.backend.api.algorithms;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.devices.Device;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public abstract class AbstractBaseAlgorithm<FinalType extends Algorithm<FinalType>> implements Algorithm<FinalType>
{
    /**
     *  This is the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     */
    private final String _name;

    protected final Map< Class< Device<?> >, ImplementationFor< Device<?> >> _implementations = new HashMap<>();

    public AbstractBaseAlgorithm(String name) { _name = name; }

    @Override
    public Tsr recursiveReductionOf(
            ExecutionCall<? extends Device<?>> call,
            Consumer<ExecutionCall<? extends Device<?>>> finalExecution
    ) {
        Device device = call.getDevice();
        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        Operation type = call.getOperation();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for ( int i=0; i<tsrs.length; i++ ) {
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
                    ExecutionCall.builder()
                            .device(device)
                            .tensors(call.getTensors())
                            .derivativeIndex(d)
                            .operation(type)
                            .build()
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
        _implementations.put(
                (Class<Device<?>>) deviceClass,
                (ImplementationFor<Device<?>>) implementation
        );
        return (FinalType) this;
    }

    @Override
    public <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass ) {
        return (ImplementationFor<D>) _implementations.get( deviceClass );
    }

    /**
     *  This method returns the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore this name is expected to be void of any spaces
     *  or non numeric and alphabetic characters.
     *
     * @return The name of this {@link Algorithm}.
     */
    public String getName() {
        return this._name;
    }
}
