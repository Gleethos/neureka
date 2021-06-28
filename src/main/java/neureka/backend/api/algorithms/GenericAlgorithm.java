package neureka.backend.api.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.autograd.DefaultADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.assembly.FunctionParser;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.dtype.NumericType;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

public class GenericAlgorithm extends AbstractBaseAlgorithm<GenericAlgorithm> {

    public GenericAlgorithm( String name, int arity, Operation type )
    {
        super( name );
        setImplementationFor(
                HostCPU.class,
                new HostImplementation(
                        call -> {
                            Function f = new FunctionBuilder(
                                                Neureka.get().context()
                                            )
                                            .build(
                                                    type,
                                                    call.getTensors().length-1,
                                                    false
                                            );

                            boolean allNumeric = call.validate()
                                                        .all( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                                        .isValid();

                            Class<?> typeClass = Stream.of(call.getTensors()).map( t -> t.getDataType().getTypeClass() ).findFirst().get();

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
                            }
                            else if (typeClass == String.class && call.getOperation().getFunction().equals("add"))
                            {
                                call
                                        .getDevice()
                                        .getExecutor()
                                        .threaded (
                                                call.getTsrOfType( Object.class, 0 ).size(),
                                                ( start, end ) -> {
                                                    for ( int i = start; i < end; i++ ) {
                                                        StringBuilder b = new StringBuilder();
                                                        for ( int ii = 1; ii < call.getTensors().length; ii++ ) {
                                                            b.append(call.getTsrOfType( Object.class, ii ).getValueAt(i));
                                                        }
                                                        call.getTsrOfType( Object.class, 0 ).setAt(i, b.toString());
                                                    }
                                                }
                                        );
                            }
                            else
                            {

                                tryExecute(call, typeClass);
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
        Function mul = Neureka.get().context().getFunction().mul();
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
            Tsr output = Tsr.of( tensors[ 1 ].getDataType(), shp );
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

    private void tryExecute( ExecutionCall<HostCPU> call, Class<?> typeClass ) {
        Method m = findMethod( call.getOperation().getFunction(), typeClass );
        if ( m == null ) {
            if ( call.getOperation().getOperator().equals("+") ) m = findMethod( "plus", typeClass );
            else if ( call.getOperation().getOperator().equals("-") ) m = findMethod( "minus", typeClass );
            else if ( call.getOperation().getOperator().equals("*") ) m = findMethod( "times", typeClass );
            else if ( call.getOperation().getOperator().equals("*") ) m = findMethod( "multiply", typeClass );
            else if ( call.getOperation().getOperator().equals("*") ) m = findMethod( "mul", typeClass );
            else if ( call.getOperation().getOperator().equals("%") ) m = findMethod( "mod", typeClass );
        }
        Method finalMethod = m;
        call
                .getDevice()
                .getExecutor()
                .threaded (
                        call.getTsrOfType( Object.class, 0 ).size(),
                        ( start, end ) -> {
                            Object[] inputs = new Object[ call.getTensors().length - 1 ];
                            for ( int i = start; i < end; i++ ) {
                                for ( int ii = 0; ii < inputs.length; ii++ ) {
                                    inputs[ ii ] = call.getTsrOfType( Object.class, 1 + ii ).getValueAt(i);
                                }
                                call.getTsrOfType( Object.class, 0 ).setAt(i, tryExecute(finalMethod, inputs, 0));
                            }
                        }
                );
    }

    private static Object tryExecute( Method m, Object[] args, int offset ) {
        if ( offset == args.length - 1 ) return args[offset];
        else {
            try {
                args[offset + 1] = m.invoke(args[offset], args[offset + 1]);
            } catch ( Exception e ) {
                e.printStackTrace();
                return null;
            }
            return tryExecute( m, args, offset + 1 );
        }
    }

    private static Method findMethod(String name, Class<?> typeClass) {
        try {
            Method m = typeClass.getMethod(name, typeClass);
            return m;
        } catch ( SecurityException e ) {
            e.printStackTrace();
        }
        catch ( NoSuchMethodException e ) {
            e.printStackTrace();
        } finally {
            Method[] methods = typeClass.getDeclaredMethods();
            Method currentBest = null;
            double currentScore = 0;
            for ( Method m : methods ) {
                int numberOfParams = m.getParameterCount();
                Class<?> type = (numberOfParams == 0) ? null : m.getParameterTypes()[0];
                if ( numberOfParams == 1 && type == typeClass ) {
                    double score = FunctionParser.similarity( m.getName(), name );
                    if ( score > currentScore ) {
                        currentBest = m;
                        currentScore = score;
                    }
                }
            }
            if ( currentScore > 0.5 ) return currentBest;
        }
        return null;

    }

}
