package neureka.calculus;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.Operation;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.operations.JunctionUtil;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionConstant;
import neureka.devices.Device;
import neureka.common.utility.Messages;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 *  This is a utility class which helps with orchestrating the execution of classical operations
 *  from calculus like the operators '*', '-', '+', '/', as well as linear operations
 *  like matrix multiplication, broadcasting and convolution.
 *  This orchestration refers to the way an {@link ExecutionCall} alongside its caller, a {@link Function},
 *  should be handles to produce a correct result.
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
        Tsr<?>[] inputs = call.getTensors();
        Device<?> device = call.getDevice();
        int j = call.getJ();
        assert call.getValOf( Arg.DerivIdx.class ) == -1;

        Tsr<?>[] tensors;
        if ( operation.isIndexer() ) tensors = new Tsr[ 1 + inputs.length ];
        else tensors = new Tsr[ 1 + nodes.length ];

        if ( operation.isIndexer() ) {
            for ( int i = 1; i < tensors.length; i++ ) tensors[ i ] = nodes[ 0 ].execute( inputs, i - 1 );
        } else if (
                !isFlat && j < 0 && (
                        operation.isOperator() || operation.supportsAlgorithm(Activation.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            tensors = srcActivation(inputs, j, -1, 0, nodes);
            String asStr = operation.stringify(
                    IntStream.range(0, nodes.length).mapToObj( i -> "I[" + i + "]" ).toArray(String[]::new)
            );
            return new FunctionBuilder( Neureka.get().backend() ).build( asStr, isDoingAD ).execute( tensors );
        } else
            tensors = srcActivation( inputs, j, -1, 1, nodes );

        CalcUtil.recursiveExecution(
                ExecutionCall.of( tensors )
                        .andArgs( Arg.DerivIdx.of(-1) )
                        .running( operation )
                        .on( device ),
                executor
        );
        if ( tensors[ 0 ] == null )
            _LOG.warn("Executing operation '"+operation.getFunction()+"' did not yield a proper return value.");

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
        for ( Tsr<?> t : tensors ) if ( t != null && !t.isVirtual() ) allVirtual = false;
        if ( allVirtual ) {
            int index = -1;
            for ( int i = 0; i < tensors.length; i++ ) {
                double value = ( tensors[ i ] == null ) ? 0.0 : tensors[ i ].value64( 0 );
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
        Supplier<Tsr<?>> actor = () -> {
            Tsr<?>[] inputs = call.getTensors();
            Device<?> device = call.getDevice();
            int d = call.getValOf( Arg.DerivIdx.class );
            int j = call.getJ();
            assert d >= 0;

            Tsr<?>[] tensors;
            if ( operation.isIndexer() ) tensors = new Tsr[ 1 + inputs.length ];
            else tensors = new Tsr[ 1 + nodes.length ];

            // Chain-rule (forward AutoDiff):
            // inner times outer means:
            // first derive source!
            // like so:
            if ( operation.isIndexer() ) {
                for ( int i = 1; i < tensors.length; i++ ) {
                    tensors[ i ] = nodes[ 0 ].executeDerive( inputs, d, i - 1 );
                }
            } else {
                for ( int i = 1; i < tensors.length; i++ ) {
                    tensors[ i ] =
                            ( j >= 0 )
                                    ? nodes[ i - 1 ].executeDerive( inputs, d, j )
                                    : nodes[ i - 1 ].executeDerive( inputs, d );
                }
            }
            //...then add them all together! (is possible because of linearity...)
            Tsr<?> inner;
            if ( tensors.length > 2 ) {// Optimization: Finds index of "1.0" among otherwise all "0.0" virtual tensors!
                int index = _indexOfFoundDerivative( tensors );
                if ( index >= 0 ) inner = tensors[ index ];
                else {
                    // Optimization above did not apply, so we accumulate all the derivatives!
                    CalcUtil.recursiveExecution(
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
            if ( operation.isIndexer() ) { // Indexer pass an index j of course!
                for ( int i = 1; i < tensors.length; i++ ) {
                    tensors[ i ] = nodes[ 0 ].execute( inputs, i - 1 ); // i - 1 := j
                }
            } else {
                for ( int i = 1; i < tensors.length; i++ ) {
                    tensors[ i ] = ( j >= 0 ) ? nodes[ i - 1 ].execute( inputs, j ) : nodes[ i - 1 ].execute( inputs );
                }
            }
            //...get derivative index within src list:
            for ( int i = 0; i < nodes.length; i++ ) {
                if ( nodes[ i ].dependsOn( d ) && !operation.isIndexer() ) {
                    d = i;
                    break;
                }
            }
            // Use those tensors for the outer derivative:
            CalcUtil.recursiveExecution(
                    ExecutionCall.of( tensors )
                            .andArgs( Arg.DerivIdx.of( d ) )
                            .running( operation )
                            .on( device ),
                    executor
            );
            // At the end:
            //...multiply inner times outer: ( if inner is not 1 entirely... )
            if ( !( ( inner.isVirtual() || inner.size() == 1 ) && inner.value64( 0 ) == 1.0 ) ) {
                tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
                CalcUtil.recursiveExecution(
                        ExecutionCall.of( tensors )
                                .andArgs( Arg.DerivIdx.of( -1 ) )
                                .running( Neureka.get().backend().getOperation("*") )
                                .on( device ),
                        null
                );
            } // done!
            return tensors[ 0 ];
        };

        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Tsr<?> out = null;
        for ( int i = 0; i < nodes.length; i++ ) { // constants need to be figured out!
            int di = ( nodes[ i ].dependsOn( d ) ) ? i : -1;
            if ( di >= 0 ) {
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
        }
        return out;
    }

    public static void recursiveExecution(
            ExecutionCall<? extends Device<?>> executionCall,
            RecursiveExecutor executor
    ) {
        executionCall = executionCall.getAlgorithm().prepare( executionCall );
        for ( Tsr<?> t : executionCall.getTensors() ) {
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );
        }
        _recursiveReductionOf(
                executionCall,
                call -> {
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( t == null ) throw new IllegalArgumentException(
                                "Device arguments may not be null!\n" +
                                        "One or more tensor arguments within the given ExecutionCall instance is null."
                        );
                    }
                    call = (ExecutionCall<? extends Device<?>>) ExecutionCall.of( call.getTensors() )
                                                                    .andArgs( Arg.DerivIdx.of( call.getValOf( Arg.DerivIdx.class ) ) )
                                                                    .running( call.getOperation() )
                                                                    .on( call.getDevice() )
                                                                    .forDeviceType( call.getDevice().getClass() );
                    Device<?> device = call.getDevice();
                    device.approve( call );
                    call.getTensors()[ 0 ].setIsVirtual( false );

                    Algorithm<?> algorithm = call.getAlgorithm();
                    if ( algorithm == null ) {
                        String message = Messages.Devices.couldNotFindSuitableAlgorithmFor( device.getClass() );
                        _LOG.error( message );
                        throw new IllegalStateException( message );
                    } else {
                        ImplementationFor<Device<?>> implementation = algorithm.getImplementationFor( device );
                        if ( implementation == null ) {
                            String message = Messages.Devices.couldNotFindSuitableImplementationFor( algorithm, device.getClass() );
                            _LOG.error( message );
                            throw new IllegalStateException( message );
                        }
                        else implementation.run( (ExecutionCall<Device<?>>) call );
                    }
                },
                executor
        );
    }

    /**
     *  The following method can be used to split one big execution call into many
     *  grouped execution calls which will be executed recursively.
     *  This method receives a the call which ought to be broken down as well as two lambdas
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
            final Consumer<ExecutionCall<? extends Device<?>>> finalExecution,
            final RecursiveExecutor executor
    ) {
        Device<Object> device = call.getDeviceFor(Object.class);
        Tsr<Object>[] tensors = (Tsr<Object>[]) call.getTensors();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Consumer<Tsr<Object>>[] rollbacks = new Consumer[ tensors.length ];
        for ( int i = 0; i < tensors.length; i++ ) {
            if ( tensors[ i ] != null && !tensors[ i ].isOutsourced() ) {
                try {
                    device.store( tensors[ i ] );
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
            finalExecution.accept(
                    ExecutionCall.of( call.getTensors() )
                            .andArgs( Arg.DerivIdx.of(d) )
                            .running( type )
                            .on( device )
            );
        }
        else return result;

        for ( int i = 0; i < tensors.length; i++ ) {
            if ( tensors[ i ] != null && !tensors[ i ].isUndefined() ) rollbacks[ i ].accept(tensors[ i ]);
        }
        return tensors[ 0 ];
    }

    /**
     *  This method performs a classical execution of a {@link Function} alongside an array of provided
     *  arguments and an offset used to make room in the output array returned by this method.
     */
    @Contract( pure = true )
    public static Tsr<?>[] srcActivation(
            Tsr<?>[] inputs, int j, int d, int offset, Function[] src
    ) {
        int[] tempShape = null;
        Tsr<?>[] tensors = new Tsr[ src.length + offset ];
        for ( int i = offset; i < tensors.length; i++ ) {//constants need to be figured out!
            if ( !( src[ i - offset ] instanceof FunctionConstant ) ) {
                if ( d < 0 ) // Not deriving this!
                    tensors[ i ] =
                            ( j >= 0 )
                                    ? src[ i - offset ].execute( inputs, j )
                                    : src[ i - offset ].execute( inputs );
                else // ...deriving at specified index...
                    tensors[ i ] =
                            ( j >= 0 )
                                    ? src[ i - offset ].executeDerive( inputs, d, j )
                                    : src[ i - offset ].executeDerive( inputs, d );

                tempShape = ( tempShape == null ? tensors[ i ].getNDConf().shape() : tempShape );
            }
        }
        for ( int i = offset; i < tensors.length; i++ ) {
            if ( tensors[ i ] == null )
                tensors[ i ] =
                        ( j < 0 )
                            ? Tsr.of( tempShape, ((FunctionConstant) src[ i - offset ]).value() )
                            : Tsr.of( tempShape, src[ i - offset ].call(new double[]{}, j) );
        }
        return tensors;
    }

}
