package neureka.backend.api.template.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.fun.ADAgentSupplier;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.ExecutionPreparation;
import neureka.backend.api.Result;
import neureka.backend.main.implementations.CPUImplementation;
import neureka.backend.main.memory.MemUtil;
import neureka.backend.main.operations.linear.MatMul;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.calculus.assembly.ParseUtil;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.stream.Stream;

public final class FallbackAlgorithm extends AbstractDeviceAlgorithm<FallbackAlgorithm>
implements ExecutionPreparation, ADAgentSupplier
{

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
                            Function f = new FunctionParser(
                                                    Neureka.get().backend()
                                                )
                                                .parse(
                                                        type,
                                                        call.arity() - 1,
                                                        false
                                                );

                            boolean allNumeric = call.validate()
                                                        .all( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                                        .isValid();

                            Class<?> typeClass = Stream.of( call.inputs() )
                                                        .map( t -> t.getDataType().getRepresentativeType() )
                                                        .findFirst()
                                                        .get();
                            if ( allNumeric )
                            {
                                double[] inputs = new double[ call.arity()-1 ];
                                call.getDevice()
                                    .getExecutor()
                                    .threaded(
                                        call.input( Number.class, 0 ).size(),
                                        ( start, end ) -> {
                                            for ( int i = start; i < end; i++ ) {
                                                for ( int ii = 0; ii < inputs.length; ii++ ) {
                                                    inputs[ ii ] = call.input( Number.class, 1 + ii ).at( i ).get().doubleValue();
                                                }
                                                call.input( Number.class, 0 ).getUnsafe().getDataAs( double[].class )[ i ] = f.call( inputs );
                                            }
                                        }
                                    );
                            }
                            else if ( typeClass == String.class && call.getOperation().getIdentifier().equals("add") )
                            {
                                call.getDevice()
                                    .getExecutor()
                                    .threaded(
                                        call.input( Object.class, 0 ).size(),
                                        ( start, end ) -> {
                                            for ( int i = start; i < end; i++ ) {
                                                StringBuilder b = new StringBuilder();
                                                for (int ii = 1; ii < call.arity(); ii++ ) {
                                                    b.append(call.input( Object.class, ii ).getItemAt(i));
                                                }
                                                setAt( call.input( Object.class, 0 ), i, b.toString() );
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
        for ( Tsr<?> t : call.inputs() ) {
            if ( t != null ) {
                if ( shape == null ) shape = t.getNDConf().shape();
                else if ( !Arrays.equals(shape, t.getNDConf().shape()) ) return 0.0f;
            }
        }
        if ( call.getOperation().getClass() == MatMul.class ) return 0;
        return 0.5f;
    }

    @Override
    public ADAgent supplyADAgentFor( Function function, ExecutionCall<? extends Device<?>> call )
    {
        Tsr<?> derivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
        Function mul = Neureka.get().backend().getFunction().mul();
        if ( derivative != null ) {
            return ADAgent.of( derivative )
                    .withAD( target -> mul.execute( target.error(), derivative ) );
        }
        Tsr<?> localDerivative = MemUtil.keep( call.inputs(), () -> function.executeDerive( call.inputs(), call.getDerivativeIndex() ) );
        localDerivative.getUnsafe().setIsIntermediate( false );
        return ADAgent.of( localDerivative )
                    .withAD( target -> mul.execute( target.error(), localDerivative ) );
        // TODO: Maybe delete local derivative??
    }

    public Tsr<?> dispatch( Function caller, ExecutionCall<? extends Device<?>> call ) {
        return CalcUtil.executeFor( caller, call, null );
    }

    @Override
    public ExecutionCall<? extends Device<?>> prepare( ExecutionCall<? extends Device<?>> call )
    {
        Device<Object> device = call.getDeviceFor(Object.class);
        if ( call.input( 0 ) == null ) // Creating a new tensor:
        {
            int[] shp = call.input( 1 ).getNDConf().shape();
            Tsr<Object> output = (Tsr<Object>) Tsr.of( call.input( 1 ).getDataType(), shp )
                                                    .getUnsafe()
                                                    .setIsIntermediate(true);
            output.setIsVirtual( false );
            try {
                device.store( output );
            } catch ( Exception e ) {
                e.printStackTrace();
            }
            call.setInput( 0, output );
        }
        return call;
    }

    private void _tryExecute( ExecutionCall<CPU> call, Class<?> typeClass ) {
        Method m = _findMethod( call.getOperation().getIdentifier(), typeClass );
        if ( m == null ) {
            switch (call.getOperation().getOperator()) {
                case "+": m = _findMethod("plus", typeClass);break;
                case "-": m = _findMethod("minus", typeClass);break;
                case "*":
                    m = _findMethod("times", typeClass);
                    if ( m == null ) m = _findMethod("multiply", typeClass);
                    if ( m == null ) m = _findMethod("mul", typeClass);
                    break;
                case "%": m = _findMethod("mod", typeClass);break;
            }
        }
        Method finalMethod = m;
        call
            .getDevice()
            .getExecutor()
            .threaded(
                    call.input( Object.class, 0 ).size(),
                    ( start, end ) -> {
                        Object[] inputs = new Object[ call.arity() - 1 ];
                        for ( int i = start; i < end; i++ ) {
                            for ( int ii = 0; ii < inputs.length; ii++ ) {
                                inputs[ ii ] = call.input( Object.class, 1 + ii ).getItemAt(i);
                            }
                            setAt( call.input( Object.class, 0 ), i, _tryExecute(finalMethod, inputs, 0));
                        }
                    }
            );
    }

    private static void setAt( Tsr<Object> t, int i, Object o ) {
        t.getUnsafe().setDataAt( t.getNDConf().indexOfIndex( i ), o );
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

    @Override
    public AutoDiffMode autoDiffModeFrom(ExecutionCall<? extends Device<?>> call ) { return AutoDiffMode.FORWARD_AND_BACKWARD; }

    @Override
    public Result execute(Function caller, ExecutionCall<? extends Device<?>> call) {
        return Result.of(this.dispatch(caller, call)).withAutoDiff(this);
    }
}
