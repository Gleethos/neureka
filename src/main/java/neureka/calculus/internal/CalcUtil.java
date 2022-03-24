package neureka.calculus.internal;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.memory.MemUtil;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionConstant;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 *  This is a utility class which helps with orchestrating the execution of classical
 *  mathematical operations like the operators '*', '-', '+', '/', as well as linear operations
 *  like matrix multiplication, broadcasting and convolution.
 *  This orchestration refers to the way an {@link ExecutionCall} alongside its caller, a {@link Function},
 *  should be handled to produce a correct result.
 */
public class CalcUtil
{
    private static final Logger _LOG = LoggerFactory.getLogger( CalcUtil.class );

    @Contract( pure = true )
    public static Tsr<?> defaultRecursiveExecution(
            final Function caller,
            final ExecutionCall<? extends Device<?>> call
    ) {
        return executeFor( caller, call, null );
    }

    @Contract( pure = true )
    public static Tsr<?> executeFor(
            final Function caller,
            final ExecutionCall<? extends Device<?>> call,
            final RecursiveExecutor executor
    ) {
        Function[] nodes = caller.getSubFunctions().toArray(new Function[0]);
        Operation operation = caller.getOperation();
        boolean isFlat = caller.isFlat();
        boolean isDoingAD = caller.isDoingAD();
        if ( call.getValOf( Arg.DerivIdx.class ) < 0 )
            return _deepActivation( call, nodes, operation, isFlat, isDoingAD, executor );
        else
            return _deepDerivative( call, nodes, operation, executor );
    }

    @Contract( pure = true )
    private static Tsr<?> _deepActivation(
            final ExecutionCall<? extends Device<?>> call,
            final Function[] nodes,
            final Operation operation,
            final boolean isFlat,
            final boolean isDoingAD,
            final RecursiveExecutor executor
    ) {
        Device<?> device = call.getDevice();
        int j = call.getValOf( Arg.VarIdx.class );
        assert call.getValOf( Arg.DerivIdx.class ) == -1;

        Tsr<?>[] tensors =
                    operation.isIndexer()
                        ? new Tsr[ 1 + call.arity() ]
                        : new Tsr[ 1 + nodes.length  ];

        if ( operation.isIndexer() )
            for ( int i = 1; i < tensors.length; i++ )
                tensors[ i ] = nodes[ 0 ].execute( call.inputs(), i - 1 );
        else
            if (
                !isFlat && j < 0 && (
                    operation.isOperator()
                            ||
                    operation.supportsAlgorithm(Activation.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            tensors = srcActivation( call.inputs(), j, -1, 0, nodes );
            String asStr = operation.stringify(
                                                IntStream.range( 0, nodes.length )
                                                         .mapToObj( i -> "I[" + i + "]" )
                                                         .toArray(String[]::new)
                                            );
            Tsr<?>[] finalTensors = tensors;
            Tsr<?> result = MemUtil.keep( tensors, () -> new FunctionBuilder( Neureka.get().backend() ).build( asStr, isDoingAD ).execute(finalTensors) );
            for ( int i = 1; i < tensors.length; i++ )
                _deleteIfNotIn( call.inputs(), tensors[ i ] );

            return result;
        }
        else
            tensors = srcActivation( call.inputs(), j, -1, 1, nodes );

        tensors[0] = CalcUtil.recursiveExecution(
                                ExecutionCall.of( tensors )
                                                .andArgs( call.allMetaArgs() )
                                                .running( operation )
                                                .on( device )
                                                .withArgs( Arg.DerivIdx.of(-1) )
                                                .withArgs( Arg.VarIdx.of(-1) ),
                                executor
                            );

        if ( tensors[ 0 ] == null ) // TODO: Fix this for 'left_inline'!!!
            _LOG.debug("Executing operation '"+operation.getIdentifier()+"' did not yield a proper return value.");

        return ( tensors[ 0 ] == null ? tensors[ 1 ] : tensors[ 0 ] );
    }

    /**
     *  This method return the index of the tensor
     *  in the given tensor array which is virtual and contains "1.0".
     *  However, if not all tensors are virtual or their values are not all "0.0" except one
     *  whose value is "1.0" then it returns -1, because the optimization cannot
     *  be made...
     *
     * @param tensors An array of tensors which ought to be analyzed.
     * @return The index of the tensor whose value is "1.0" (if all others are "0.0"), otherwise : -1
     */
    @Contract( pure = true )
    private static int _indexOfFoundDerivative( Tsr<?>[] tensors )
    {
        boolean allVirtual = true;
        for ( Tsr<?> t : tensors )
            if ( t != null && !t.isVirtual() ) allVirtual = false;

        if ( allVirtual ) {
            int index = -1;
            for ( int i = 0; i < tensors.length; i++ ) {
                double value = ( tensors[ i ] == null ? 0.0 : tensors[ i ].getValueAs( double[].class )[ 0 ] );
                if ( value == 1.0 ) {
                    if ( index >= 0 ) return -1;
                    index = i;
                }
                else if ( value != 0.0 ) return -1;
            }
            return index;
        }
        return -1;
    }

    @Contract( pure = true )
    private static Tsr<?> _deepDerivative(
            final ExecutionCall<? extends Device<?>> call,
            final Function[] nodes,
            final Operation operation,
            final RecursiveExecutor executor
    ) {
        Supplier<Tsr<?>> actor = () ->
                MemUtil.keep( call.inputs(), () -> {
                    Device<?> device = call.getDevice();
                    int d = call.getValOf( Arg.DerivIdx.class );
                    int j = call.getValOf( Arg.VarIdx.class );
                    assert d >= 0;

                    Tsr<?>[] tensors;
                    if ( operation.isIndexer() ) tensors = new Tsr[ 1 + call.arity() ];
                    else tensors = new Tsr[ 1 + nodes.length ];

                    // Chain-rule (forward AutoDiff):
                    // inner times outer means:
                    // first derive source!
                    // like so:
                    if ( operation.isIndexer() )
                        for ( int i = 1; i < tensors.length; i++ )
                            tensors[ i ] = nodes[ 0 ].executeDerive( call.inputs(), d, i - 1 );
                    else
                        for ( int i = 1; i < tensors.length; i++ )
                            tensors[ i ] =
                                        j >= 0
                                            ? nodes[ i - 1 ].executeDerive( call.inputs(), d, j )
                                            : nodes[ i - 1 ].executeDerive( call.inputs(), d    );

                    //...then add them all together! (is possible because of linearity...)
                    Tsr<?> inner;
                    if ( tensors.length > 2 ) {// Optimization: Finds index of "1.0" among otherwise all "0.0" virtual tensors!
                        int index = _indexOfFoundDerivative( tensors );
                        if ( index >= 0 ) inner = tensors[ index ];
                        else {
                            // Optimization above did not apply, so we accumulate all the derivatives!
                            tensors[0] = CalcUtil.recursiveExecution(
                                                    ExecutionCall.of( tensors )
                                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                                            .running( Neureka.get().backend().getOperation("+") )
                                                            .on( device ),
                                                    JunctionUtil::forAdditions
                                            );
                            inner = tensors[ 0 ];//-> this is now the inner derivative!
                        }
                    }
                    else inner = tensors[ 1 ];

                    tensors[ 0 ] = null;
                    //...then activate (No differentiation!) the source like so:
                    if ( operation.isIndexer() ) // Indexer pass an index j of course!
                        for ( int i = 1; i < tensors.length; i++ )
                            tensors[ i ] = nodes[ 0 ].execute( call.inputs(), i - 1 ); // i - 1 := j
                    else
                        for ( int i = 1; i < tensors.length; i++ )
                            tensors[ i ] =
                                    j >= 0
                                        ? nodes[ i - 1 ].execute( call.inputs(), j )
                                        : nodes[ i - 1 ].execute( call.inputs() );

                    //...get derivative index within src list:
                    for ( int i = 0; i < nodes.length; i++ )
                        if ( nodes[ i ].dependsOn( d ) && !operation.isIndexer() ) {
                            d = i;
                            break;
                        }

                    // Use those tensors for the outer derivative:
                    tensors[0] = CalcUtil.recursiveExecution(
                                            ExecutionCall.of( tensors )
                                                    .andArgs( Arg.DerivIdx.of( d ) )
                                                    .running( operation )
                                                    .on( device ),
                                            executor
                                    );
                    // At the end:
                    //...multiply inner times outer: ( if inner is not 1 entirely... )
                    if ( !( ( inner.isVirtual() || inner.size() == 1 ) && inner.getValueAs( double[].class )[ 0 ] == 1.0 ) ) {
                        tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
                        tensors[0] = CalcUtil.recursiveExecution(
                                                ExecutionCall.of( tensors )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("*") )
                                                        .on( device ),
                                                null
                                        );
                        for ( int i = 1; i < tensors.length; i++ )
                            _deleteIfNotIn( call.inputs(), tensors[ i ] );
                    }
                    // done!

                    _delete( inner );

                    return tensors[ 0 ];
                });

        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Tsr<?> out = null;
        for ( int i = 0; i < nodes.length; i++ )
        {
            // constants need to be figured out!
            int di = ( nodes[ i ].dependsOn( d ) ? i : -1 );
            if ( di >= 0 )
                if ( out == null ) out = actor.get();
                else
                    CalcUtil.recursiveExecution(
                            ExecutionCall.of( null, actor.get(), out )
                                    .andArgs( Arg.DerivIdx.of( -1 ) )
                                    .running( Neureka.get().backend().getOperation( "+" ) )
                                    .on( device ),
                            null
                    );
        }
        return out;
    }

    private static void _deleteIfNotIn( Tsr<?>[] array, Tsr<?> tensor ) {
        if ( Neureka.get().settings().debug().isDeletingIntermediateTensors() ) {
            for ( int i = 1; i < array.length; i++ )
                if ( array[i] == tensor ) return;

            if ( !tensor.isDeleted() ) tensor.getUnsafe().delete();
        }
    }

    private static void _delete( Tsr<?> tensor ) {
        if ( Neureka.get().settings().debug().isDeletingIntermediateTensors() )
            if ( !tensor.isDeleted() ) tensor.getUnsafe().delete();
    }

    public static Tsr<?> recursiveExecution(
            ExecutionCall<? extends Device<?>> executionCall,
            RecursiveExecutor executor
    ) {
        executionCall = executionCall.getAlgorithm().prepare( executionCall );

        for ( Tsr<?> t : executionCall.inputs() )
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );

        Tsr<?> result =
                _recursiveReductionOf(
                    executionCall,
                    call -> {
                        for ( Tsr<?> t : call.inputs() )
                            if ( t == null ) throw new IllegalArgumentException(
                                    "Device arguments may not be null!\n" +
                                            "One or more tensor arguments within the given ExecutionCall instance is null."
                            );

                        call = ExecutionCall.of( call.inputs() )
                                            .andArgs( call.allMetaArgs() )
                                            .running( call.getOperation() )
                                            .on( call.getDevice() );

                        Device<?> device = call.getDevice();
                        device.approve( call );

                        Algorithm<?> algorithm = call.getAlgorithm();
                        if ( algorithm == null ) {
                            String message = _couldNotFindSuitableAlgorithmFor( device.getClass() );
                            _LOG.error( message );
                            throw new IllegalStateException( message );
                        } else {
                            ImplementationFor<Device<?>> implementation = algorithm.getImplementationFor( device );
                            if ( implementation == null ) {
                                String message = _couldNotFindSuitableImplementationFor( algorithm, device.getClass() );
                                _LOG.error( message );
                                throw new IllegalStateException( message );
                            }
                            else implementation.run( (ExecutionCall<Device<?>>) call );
                        }
                        return call.input( 0 );
                    },
                    executor
                );

        //assert result == executionCall.tensor(0);
        return result;
    }

    /**
     *  The following method can be used to split one big execution call into many
     *  grouped execution calls which will be executed recursively.
     *  This method receives a call which ought to be broken down as well as two lambdas
     *  which contain implementations to perform this task.
     *  The first lambda, namely {@param finalExecution}, will be called at the end of the
     *  recursion dive, whereas the second lambda {@param executor} will be called for
     *  every recursive call in order to perform the grouping.
     *  The {@param executor} will actually receive the recursive call as lambda, which
     *  then may or may not be called by implementations of the lambda...
     *
     * @param call The {@link ExecutionCall} whose arguments ought to be executed in groups.
     * @param finalExecution The actual execution whose implementation is provided by the caller.
     * @param executor The traversing algorithm, which decides how to group arguments and when
     *                 the {@param finalExecution} ought to be called.
     *
     * @return The execution result of the provided {@param call}.
     */
    @Contract( pure = true )
    private static Tsr<?> _recursiveReductionOf(
            final ExecutionCall<? extends Device<?>> call,
            final java.util.function.Function<ExecutionCall<? extends Device<?>>, Tsr<?>> finalExecution,
            final RecursiveExecutor executor
    ) {
        Device<Object> device = call.getDeviceFor(Object.class);
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Consumer<Tsr<?>>[] rollbacks = new Consumer[ call.arity() ];
        for (int i = 0; i < call.arity(); i++ )
            if ( call.input( i ) != null && !call.input( i ).isOutsourced() ) {
                try {
                    device.store( call.input( i ) );
                } catch ( Exception e ) {
                    e.printStackTrace();
                }

                rollbacks[ i ] = tensor -> {
                                        try {
                                            device.restore( (Tsr<Object>) tensor );
                                        } catch ( Exception e ) {
                                            e.printStackTrace();
                                        }
                                    };
            }
            else
                rollbacks[ i ] = t -> {};
        /*
            Below is the core lambda of recursive preprocessing
            which is defined for each Algorithm individually :
         */
        Tsr<?> result = null;
        if ( executor != null )
            result = executor.execute( // This is where the recursion occurs:
                                call,
                                innerCall -> // This lambda performs the recursive call, implementations decide if they want to dive deeper.
                                        _recursiveReductionOf( innerCall, finalExecution, executor )
                            );

        if ( result == null ) {
            call.setInput( 0,
                    finalExecution.apply(
                        ExecutionCall.of( call.inputs() )
                                .andArgs(call.allMetaArgs())
                                .running(type)
                                .on(device)
                                .withArgs( Arg.DerivIdx.of(d) )
                    )
               );
        } else
            return result;

        for (int i = 0; i < call.arity(); i++ )
            if ( call.input( i ) != null && !call.input( i ).isUndefined() )
                rollbacks[ i ].accept( call.input( i ) );

        return call.input( 0 );
    }

    /**
     *  This method performs a classical execution of a {@link Function} alongside an array of provided
     *  arguments and an offset used to make room in the output array returned by this method.
     */
    @Contract( pure = true )
    public static Tsr<?>[] srcActivation(
            Tsr<?>[] inputs, int j, int d, int offset, Function[] src
    ) {
        return MemUtil.keep( inputs, () ->
        {
            int[] tempShape = null;
            Class<?> tempType = null;
            Tsr<?>[] tensors = new Tsr[ src.length + offset ];
            for ( int i = offset; i < tensors.length; i++ ) {//constants need to be figured out!
                if ( !( src[ i - offset ] instanceof FunctionConstant ) ) {
                    if ( d < 0 ) // Not deriving this!
                        tensors[ i ] =
                                    j >= 0
                                        ? src[ i - offset ].execute( inputs, j )
                                        : src[ i - offset ].execute( inputs );
                    else // ...deriving at specified index...
                        tensors[ i ] =
                                    j >= 0
                                        ? src[ i - offset ].executeDerive( inputs, d, j )
                                        : src[ i - offset ].executeDerive( inputs, d );

                    tempShape = ( tempShape == null ? tensors[ i ].getNDConf().shape() : tempShape );
                    tempType  = ( tempType  == null ? tensors[ i ].getValueClass()     : tempType  );
                }
            }
            for ( int i = offset; i < tensors.length; i++ )
                if ( tensors[ i ] == null )
                    tensors[ i ] =
                                j < 0
                                    ? Tsr.of( tempType, tempShape, ((FunctionConstant) src[ i - offset ]).value() ).getUnsafe().setIsIntermediate( true )
                                    : Tsr.of( tempType, tempShape, src[ i - offset ].call(new double[]{}, j) ).getUnsafe().setIsIntermediate( true );
            return tensors;
        });
    }


    private static String _couldNotFindSuitableAlgorithmFor(Class<?> type ) {
        return LogUtil.format(
                "No suitable '"+ Algorithm.class.getSimpleName()+"' found for device of type '{}'.",
                type.getSimpleName()
        );
    }

    private static String _couldNotFindSuitableImplementationFor(
            Algorithm<?> algorithm,
            Class<?> type
    ) {
        return LogUtil.format(
                "No suitable implementation found for algorithm '{}' and device type '{}'.",
                algorithm.getName(),
                type.getSimpleName()
        );
    }

}
