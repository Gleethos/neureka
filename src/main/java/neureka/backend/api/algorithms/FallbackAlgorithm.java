package neureka.backend.api.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.backend.standard.operations.linear.MatMul;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.assembly.ParseUtil;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

public final class FallbackAlgorithm extends AbstractBaseAlgorithm<FallbackAlgorithm> {

    private static final Logger _LOG = LoggerFactory.getLogger(FallbackAlgorithm.class);

    public FallbackAlgorithm( String name, int arity, Operation type )
    {
        super( name );
        setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity( arity )
                    .andImplementation(
                        call -> {
                            Function f = new FunctionBuilder(
                                                    Neureka.get().backend()
                                                )
                                                .build(
                                                        type,
                                                        call.getTensors().length - 1,
                                                        false
                                                );

                            boolean allNumeric = call.validate()
                                                        .all( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                                        .isValid();

                            Class<?> typeClass = Stream.of( call.getTensors() )
                                                        .map( t -> t.getDataType().getTypeClass() )
                                                        .findFirst()
                                                        .get();
                            if ( allNumeric )
                            {
                                double[] inputs = new double[ call.getTensors().length-1 ];
                                call.getDevice()
                                    .getExecutor()
                                    .threaded(
                                        call.getTsrOfType( Number.class, 0 ).size(),
                                        ( start, end ) -> {
                                            for ( int i = start; i < end; i++ ) {
                                                for ( int ii = 0; ii < inputs.length; ii++ ) {
                                                    inputs[ ii ] = call.getTsrOfType( Number.class, 1 + ii ).getValueAs( double[].class )[ i ];
                                                }
                                                call.getTsrOfType( Number.class, 0 ).getDataAs( double[].class )[ i ] = f.call( inputs );
                                            }
                                        }
                                    );
                            }
                            else if ( typeClass == String.class && call.getOperation().getFunction().equals("add") )
                            {
                                call.getDevice()
                                    .getExecutor()
                                    .threaded(
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
                                _tryExecute(call, typeClass);
                        }
                    )
        );
    }

    @Override
    public float isSuitableFor( ExecutionCall<? extends Device<?>> call ) {
        int[] shape = null;
        for ( Tsr<?> t : call.getTensors() ) {
            if ( t != null ) {
                if ( shape == null ) shape = t.getNDConf().shape();
                else if ( !Arrays.equals(shape, t.getNDConf().shape()) ) return 0.0f;
            }
        }
        if ( call.getOperation().getClass() == MatMul.class ) return 0;
        return 0.5f;
    }

    @Override
    public boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call ) { return true; }

    @Override
    public boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call ) { return true; }

    @Override
    public ADAgent supplyADAgentFor( Function function, ExecutionCall<? extends Device<?>> call, boolean forward )
    {
        Tsr<?> derivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
        Function mul = Neureka.get().backend().getFunction().mul();
        if ( derivative != null ) {
            return ADAgent.of( derivative )
                    .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, derivative ) )
                    .setBackward( (node, backwardError ) -> mul.execute( backwardError, derivative ) );
        }
        Tsr<?> localDerivative = CalcUtil.keep( call.getTensors(), () -> function.executeDerive( call.getTensors(), call.getDerivativeIndex() ) );
        localDerivative.getMutate().setIsIntermediate( false );
        return ADAgent.of( localDerivative )
                      .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, localDerivative ) )
                      .setBackward( (node, backwardError ) -> mul.execute( backwardError, localDerivative ) );
        // TODO: Maybe delete local derivative??
    }

    @Override
    public Tsr<?> dispatch( FunctionNode caller, ExecutionCall<? extends Device<?>> call ) {
        return CalcUtil.defaultRecursiveExecution( caller, call );
    }

    @Override
    public ExecutionCall<? extends Device<?>> prepare( ExecutionCall<? extends Device<?>> call )
    {
        Tsr<?>[] tensors = call.getTensors();
        Device<Object> device = call.getDeviceFor(Object.class);
        if ( tensors[ 0 ] == null ) // Creating a new tensor:
        {
            int[] shp = tensors[ 1 ].getNDConf().shape();
            Tsr<Object> output = (Tsr<Object>) Tsr.of( tensors[ 1 ].getDataType(), shp )
                                                    .getMutate()
                                                    .setIsIntermediate(true);
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

    private void _tryExecute( ExecutionCall<CPU> call, Class<?> typeClass ) {
        Method m = _findMethod( call.getOperation().getFunction(), typeClass );
        if ( m == null ) {
            switch (call.getOperation().getOperator()) {
                case "+": m = _findMethod("plus", typeClass);break;
                case "-": m = _findMethod("minus", typeClass);break;
                case "*":
                    m = _findMethod("times", typeClass);
                    if ( m == null) m = _findMethod("multiply", typeClass);
                    if ( m == null) m = _findMethod("mul", typeClass);
                    break;
                case "%": m = _findMethod("mod", typeClass);break;
            }
        }
        Method finalMethod = m;
        call
            .getDevice()
            .getExecutor()
            .threaded(
                    call.getTsrOfType( Object.class, 0 ).size(),
                    ( start, end ) -> {
                        Object[] inputs = new Object[ call.getTensors().length - 1 ];
                        for ( int i = start; i < end; i++ ) {
                            for ( int ii = 0; ii < inputs.length; ii++ ) {
                                inputs[ ii ] = call.getTsrOfType( Object.class, 1 + ii ).getValueAt(i);
                            }
                            call.getTsrOfType( Object.class, 0 ).setAt(i, _tryExecute(finalMethod, inputs, 0));
                        }
                    }
            );
    }

    private static Object _tryExecute( Method m, Object[] args, int offset ) {
        if ( offset == args.length - 1 ) return args[offset];
        else {
            try {
                args[offset + 1] = m.invoke(args[offset], args[offset + 1]);
            } catch ( Exception e ) {
                _LOG.debug("Failed to execute method '"+m.getName()+"'. "+e.getMessage());
                return null;
            }
            return _tryExecute( m, args, offset + 1 );
        }
    }

    private static Method _findMethod( String name, Class<?> typeClass ) {
        try {
            return typeClass.getMethod(name, typeClass);
        } catch ( SecurityException e ) {
            e.printStackTrace();
        } catch ( NoSuchMethodException e ) {
            String message =
                    "Failed finding method named '"+name+"' on instance of type '"+typeClass.getSimpleName()+"'.\n" +
                    "Cause: "+e.getMessage();
            _LOG.debug(message);
        } finally {
            Method[] methods = typeClass.getDeclaredMethods();
            Method currentBest = null;
            double currentScore = 0;
            for ( Method m : methods ) {
                int numberOfParams = m.getParameterCount();
                Class<?> type = (numberOfParams == 0) ? null : m.getParameterTypes()[0];
                if ( numberOfParams == 1 && type == typeClass ) {
                    double score = ParseUtil.similarity( m.getName(), name );
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
