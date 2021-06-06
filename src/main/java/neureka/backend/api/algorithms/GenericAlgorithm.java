package neureka.backend.api.algorithms;

import groovy.lang.Binding;
import groovy.lang.GroovyShell;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.dtype.NumericType;

import java.util.Arrays;

public class GenericAlgorithm extends AbstractBaseAlgorithm<GenericAlgorithm> {

    public GenericAlgorithm( String name, int arity, Operation type )
    {
        super( name );
        setImplementationFor(
                HostCPU.class,
                new HostImplementation(
                        call -> {
                            Function f = new FunctionBuilder(
                                                OperationContext.get()
                                            )
                                            .build(
                                                    type,
                                                    call.getTensors().length-1,
                                                    false
                                            );

                            boolean allNumeric = call.validate()
                                                        .all( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                                        .isValid();

                            if ( allNumeric )
                            {
                                double[] inputs = new double[ call.getTensors().length-1 ];
                                call
                                        .getDevice()
                                        .getExecutor()
                                        .threaded (
                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                ( start, end ) -> {
                                                    for ( int i = start; i < end; i++ ) {
                                                        for ( int ii = 0; ii < inputs.length; ii++ ) {
                                                            inputs[ ii ] = call.getTsrOfType( Number.class, 1 + ii ).value64( i );
                                                        }
                                                        call.getTsrOfType( Number.class, 0 ).value64()[ i ] = f.call( inputs );
                                                    }
                                                }
                                        );
                            } else {
                                Object[] inputs = new Object[ call.getTensors().length-1 ];
                                String expression = f.toString();
                                Binding binding = new Binding();
                                binding.setVariable("I", inputs);
                                GroovyShell shell = new GroovyShell( binding );
                                call
                                        .getDevice()
                                        .getExecutor()
                                        .threaded (
                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                ( start, end ) -> {
                                                    for ( int i = start; i < end; i++ ) {
                                                        for ( int ii = 0; ii < inputs.length; ii++ ) {
                                                            inputs[ ii ] = call.getTsrOfType( Number.class, 1+ii).getValueAt(i);
                                                        }
                                                        call.getTsrOfType( Object.class, 0 ).setAt(i, shell.evaluate( expression ));
                                                    }
                                                }
                                        );

                            }
                        },
                        arity
                )
        );
    }

    @Override
    public float isSuitableFor( ExecutionCall<? extends Device<?>> call ) {
        int[] shape = null;
        for ( Tsr<?> t : call.getTensors() ) {
            if ( shape == null ) if ( t != null ) shape = t.getNDConf().shape();
            else if ( t != null && !Arrays.equals( shape, t.getNDConf().shape() ) ) return 0.0f;
        }
        return 1.0f;
    }

    /**
     * @param call The execution call which has been routed to this implementation...
     * @return null because the default implementation is not outsourced.
     */
    @Override
    public Device<?> findDeviceFor( ExecutionCall<? extends Device<?>> call ) {
        return null;
    }

    @Override
    public boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return true;
    }

    @Override
    public boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return true;
    }

    @Override
    public ADAgent supplyADAgentFor( Function f, ExecutionCall<? extends Device<?>> call, boolean forward)
    {
        Tsr<Object> ctxDerivative = (Tsr<Object>) call.getAt("derivative");
        Function mul = OperationContext.get().getFunction().MUL();
        if ( ctxDerivative != null ) {
            return new DefaultADAgent( ctxDerivative )
                    .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                    .setBackward( (node, backwardError ) -> mul.call( new Tsr[]{ backwardError, ctxDerivative } ) );
        }
        Tsr<?> localDerivative = f.executeDerive( call.getTensors(), call.getDerivativeIndex() );
        return new DefaultADAgent( localDerivative )
                .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
    }

    @Override
    public Tsr handleInsteadOfDevice( FunctionNode caller, ExecutionCall<? extends Device<?>> call ) {
        return null;
    }

    @Override
    public Tsr<?> handleRecursivelyAccordingToArity( ExecutionCall<? extends Device<?>> call, java.util.function.Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith )
    {
        return null;
    }

    @Override
    public ExecutionCall<? extends Device<?>> instantiateNewTensorsForExecutionIn( ExecutionCall<? extends Device<?>> call )
    {
        Tsr[] tensors = call.getTensors();
        Device device = call.getDevice();
        if ( tensors[ 0 ] == null ) // Creating a new tensor:
        {
            int[] shp = tensors[ 1 ].getNDConf().shape();
            Tsr output = new Tsr( shp, tensors[ 1 ].getDataType() );
            output.setIsVirtual( false );
            try {
                device.store( output );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            tensors[ 0 ] = output;
        }
        return call;
    }

}
