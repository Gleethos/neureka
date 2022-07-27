package neureka.backend.main.internal;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ImplementationFor;
import neureka.backend.api.*;
import neureka.backend.api.fun.ExecutionPreparation;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.memory.MemUtil;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
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
public class AlgoUtil
{
    private static final Logger _LOG = LoggerFactory.getLogger( AlgoUtil.class );

    @Contract( pure = true )
    public static Tsr<?> executeFor(
            final Function caller,
            final ExecutionCall<? extends Device<?>> call,
            final RecursiveExecutor executor
    ) {
        Function[] nodes = caller.getSubFunctions().toArray(new Function[0]);
        Operation operation = caller.getOperation();
        assert call.getOperation() == operation;
        boolean isFlat = caller.isFlat();
        boolean isDoingAD = caller.isDoingAD();
        if ( call.getValOf( Arg.DerivIdx.class ) < 0 )
            return _deepActivation( call, nodes, isFlat, isDoingAD, executor );
        else
            return _deepDerivative( call, nodes,  executor );
    }

    public static Tsr<?> prepareAndExecuteRecursively(
            ExecutionCall<? extends Device<?>> executionCall,
            RecursiveExecutor executor
    ) {
        executionCall = _prepareForExecution(executionCall);
        return
            _recursiveReductiveExecutionOf( executionCall, executor );
    }

    public static ExecutionCall<? extends Device<?>> _prepareForExecution(ExecutionCall<? extends Device<?>> executionCall) {
        Algorithm currentAlgorithm = executionCall.getAlgorithm();
        if ( currentAlgorithm instanceof ExecutionPreparation )
            executionCall = ( (ExecutionPreparation) currentAlgorithm ).prepare( executionCall );

        for ( Tsr<?> t : executionCall.inputs() )
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );
        return executionCall;
    }

    public static Tsr<?> executeDeviceAlgorithm(
        ExecutionCall<? extends Device<?>> call,
        CallExecutor executor // Ignored! Only for compatibility!
    ) {

        for ( Tsr<?> t : call.inputs() )
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );

        Device<?> device = call.getDevice();
        device.approve( call );

        Algorithm algorithm = call.getAlgorithm();
        if ( algorithm == null ) {
            String message = _couldNotFindSuitableAlgorithmFor( device.getClass() );
            _LOG.error( message );
            throw new IllegalStateException( message );
        } else {
            DeviceAlgorithm<?> deviceAlgorithm = ( algorithm instanceof DeviceAlgorithm ? ((DeviceAlgorithm<?>) algorithm) : null );
            ImplementationFor<Device<?>> implementation =  ( deviceAlgorithm == null ? null : deviceAlgorithm.getImplementationFor(device) );
            if ( implementation == null ) {
                String message = _couldNotFindSuitableImplementationFor( algorithm, device.getClass() );
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
            else return implementation.run( (ExecutionCall<Device<?>>) call );
        }
    }

    public static <D extends Device<?>> ExecutionCall<D> flatten(
        Function caller, ExecutionCall<D> call
    ) {
        return _flatten( call, caller.getSubFunctions().toArray(new Function[0]) );
    }


    @Contract( pure = true )
    private static <D extends Device<?>> ExecutionCall<D> _flatten(
            ExecutionCall<D> call, Function[] src
    ) {
        ExecutionCall<D> innerCall = call.withArgs( Arg.DerivIdx.of(-1) );
        Tsr<?>[] inputs = innerCall.inputs();
        return MemUtil.keep( inputs, () ->
        {
            int[] tempShape = null;
            Class<?> tempType = null;
            Tsr<?>[] tensors = new Tsr[src.length];
            for ( int i = 0; i < tensors.length; i++ ) {//constants need to be figured out!
                if ( !( src[i] instanceof FunctionConstant ) ) {
                    tensors[ i ] = src[i].execute(innerCall);
                    tempShape = ( tempShape == null ? tensors[ i ].getNDConf().shape() : tempShape );
                    tempType  = ( tempType  == null ? tensors[ i ].getItemClass()     : tempType  );
                }
            }
            int j = innerCall.getValOf( Arg.VarIdx.class );
            for ( int i = 0; i < tensors.length; i++ )
                if ( tensors[ i ] == null )
                    tensors[ i ] =
                            j < 0
                                ? Tsr.of( tempType, tempShape, ((FunctionConstant) src[i]).value() ).getUnsafe().setIsIntermediate( true )
                                : Tsr.of( tempType, tempShape, src[i].call(new double[]{}, j)      ).getUnsafe().setIsIntermediate( true );

            return innerCall.withInputs(tensors);
        });
    }

    @Contract( pure = true )
    private static Tsr<?> _deepActivation(
            final ExecutionCall<? extends Device<?>> call,
            final Function[] nodes,
            final boolean isFlat,
            final boolean isDoingAD,
            final RecursiveExecutor executor
    ) {
        int j = call.getValOf( Arg.VarIdx.class );
        assert call.getValOf( Arg.DerivIdx.class ) == -1;

        Tsr<?>[] tensors =
                    call.getOperation().isIndexer()
                        ? new Tsr[ 1 + call.arity() ]
                        : new Tsr[ 1 + nodes.length  ];

        if ( call.getOperation().isIndexer() )
            for ( int i = 1; i < tensors.length; i++ )
                tensors[ i ] = nodes[ 0 ].execute( call.inputs(), i - 1 );
        else
            if (
                !isFlat && j < 0 && (
                    call.getOperation().isOperator()
                            ||
                    call.getOperation().supportsAlgorithm(Activation.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            tensors = _flatten( call.withArgs( Arg.VarIdx.of(j) ), nodes ).inputs();
            String asStr = call.getOperation().stringify(
                                                IntStream.range( 0, nodes.length )
                                                         .mapToObj( i -> "I[" + i + "]" )
                                                         .toArray(String[]::new)
                                            );
            Tsr<?>[] finalTensors = tensors;
            Tsr<?> result = MemUtil.keep( tensors, () -> new FunctionParser( Neureka.get().backend() ).parse( asStr, isDoingAD ).execute(finalTensors) );
            for ( int i = 1; i < tensors.length; i++ )
                _deleteIfNotIn( call.inputs(), tensors[ i ] );

            return result;
        }
        else {
            ExecutionCall<?> flattenedCall = _flatten(call.withArgs(Arg.VarIdx.of(j)), nodes);
            int numberOfInputs = flattenedCall.arity();
            boolean anyNumberOfInputs = flattenedCall.getOperation().getArity() < 0;
            int operationArity = flattenedCall.getOperation().getArity();
            if ( numberOfInputs < operationArity )
                throw new IllegalArgumentException(
                        "The number of inputs to the operation " + flattenedCall.getOperation() + " is " + numberOfInputs +
                        " but the operation requires " + operationArity + " inputs."
                );

            boolean tooManyArgs = numberOfInputs > operationArity + 1;

            if ( !tooManyArgs || anyNumberOfInputs )
                tensors = flattenedCall.withAddedInputAt(0, null).inputs();
            else
                tensors = flattenedCall.inputs();
        }
        Tsr<?> out = AlgoUtil.prepareAndExecuteRecursively(
                                call.withInputs( tensors )
                                    .withArgs( Arg.DerivIdx.of(-1), Arg.VarIdx.of(-1) ),
                                executor
                            );

        if ( out == null )
            throw new IllegalStateException("The result of the recursive execution is null!");

        return out;
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
    private static int _indexOfFoundDerivative( final Tsr<?>[] tensors )
    {
        boolean allVirtual = true;
        for ( Tsr<?> t : tensors )
            if ( t != null && !t.isVirtual() ) allVirtual = false;

        if ( allVirtual ) {
            int index = -1;
            for ( int i = 0; i < tensors.length; i++ ) {
                double value = ( tensors[ i ] == null ? 0.0 : tensors[ i ].getItemsAs( double[].class )[ 0 ] );
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
            final RecursiveExecutor executor
    ) {
        Supplier<Tsr<?>> actor = () ->
            MemUtil.keep( call.inputs(), () -> {
                final Device<?> device = call.getDevice();
                int d = call.getValOf( Arg.DerivIdx.class );
                final int j = call.getValOf( Arg.VarIdx.class );
                assert d >= 0;

                Tsr<?>[] tensors;
                if ( call.getOperation().isIndexer() ) tensors = new Tsr[ 1 + call.arity() ];
                else tensors = new Tsr[ 1 + nodes.length ];

                // Chain-rule (forward AutoDiff):
                // inner times outer means:
                // first derive source!
                // like so:
                if ( call.getOperation().isIndexer() )
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
                        tensors[0] = AlgoUtil.prepareAndExecuteRecursively(
                                                ExecutionCall.of( tensors )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("+") )
                                                        .on( device ),
                                                ElemWiseUtil::forAdditions
                                        );
                        inner = tensors[ 0 ];//-> this is now the inner derivative!
                    }
                }
                else inner = tensors[ 1 ];

                tensors[ 0 ] = null;
                //...then activate (No differentiation!) the source like so:
                if ( call.getOperation().isIndexer() ) // Indexer pass an index j of course!
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
                    if ( nodes[ i ].dependsOn( d ) && !call.getOperation().isIndexer() ) {
                        d = i;
                        break;
                    }

                // Use those tensors for the outer derivative:
                tensors[0] = AlgoUtil.prepareAndExecuteRecursively(
                                        ExecutionCall.of( tensors )
                                                .andArgs( Arg.DerivIdx.of( d ) )
                                                .running( call.getOperation() )
                                                .on( device ),
                                        executor
                                );
                // At the end:
                //...multiply inner times outer: ( if inner is not 1 entirely... )
                if ( !( ( inner.isVirtual() || inner.size() == 1 ) && inner.getItemsAs( double[].class )[ 0 ] == 1.0 ) ) {
                    tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
                    tensors[0] = AlgoUtil.prepareAndExecuteRecursively(
                                            ExecutionCall.of( tensors )
                                                    .andArgs( Arg.DerivIdx.of( -1 ) )
                                                    .running( Neureka.get().backend().getOperation("*") )
                                                    .on( device ),
                                            AlgoUtil::executeDeviceAlgorithm
                                    );
                    for ( int i = 1; i < tensors.length; i++ )
                        _deleteIfNotIn( call.inputs(), tensors[ i ] );
                }
                // done!

                _delete( inner );

                return tensors[ 0 ];
            });

        int d = call.getValOf( Arg.DerivIdx.class );
        Tsr<?> out = null;
        for ( int i = 0; i < nodes.length; i++ )
        {
            // constants need to be figured out!
            int di = ( nodes[ i ].dependsOn( d ) ? i : -1 );
            if ( di >= 0 )
                if ( out == null ) out = actor.get();
                else
                    break;
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
     * @param executor The traversing algorithm, which decides how to group arguments and when
     *                 the {@param finalExecution} ought to be called.
     *
     * @return The execution result of the provided {@param call}.
     */
    @Contract( pure = true )
    private static Tsr<?> _recursiveReductiveExecutionOf(
            final ExecutionCall<? extends Device<?>> call,
            final RecursiveExecutor executor
    ) {
        return executeOnCommonDevice(call, ()->{
             /*
                Below is the core lambda of recursive preprocessing
                which is defined for each Algorithm individually :
             */
            Tsr<?> result = null;
            if ( executor != null )
                result = executor.execute( // This is where the recursion occurs:
                        call,
                        innerCall -> // This lambda performs the recursive call, implementations decide if they want to dive deeper.
                                _recursiveReductiveExecutionOf( innerCall, executor )
                );
            return result;
        });
    }

    public static <R> R executeOnCommonDevice(ExecutionCall<?> call, Supplier<R> execution ) {
        Device<Object> device = call.getDeviceFor(Object.class);

        Consumer<Tsr<?>>[] rollbacks = new Consumer[ call.arity() ];
        for (int i = 0; i < call.arity(); i++ )
            if ( call.input( i ) != null && !call.input( i ).isOutsourced() ) {
                device.store( call.input( i ) );
                rollbacks[ i ] = tensor -> device.restore( (Tsr<Object>) tensor );
            }
            else
                rollbacks[ i ] = t -> {};

        R result = execution.get();

        if ( result == null )
            throw new IllegalStateException( "Execution of " + call + " failed!" );

        for ( int i = 0; i < call.arity(); i++ )
            if ( call.input( i ) != null && !call.input( i ).isDeleted() && !call.input( i ).isUndefined() )
                rollbacks[ i ].accept( call.input( i ) );

        return result;
    }

    private static String _couldNotFindSuitableAlgorithmFor(Class<?> type ) {
        return LogUtil.format(
                "No suitable '"+ Algorithm.class.getSimpleName()+"' found for device of type '{}'.",
                type.getSimpleName()
        );
    }

    private static String _couldNotFindSuitableImplementationFor(
            Algorithm algorithm,
            Class<?> type
    ) {
        return LogUtil.format(
                "No suitable implementation found for algorithm '{}' and device type '{}'.",
                algorithm.getName(),
                type.getSimpleName()
        );
    }

}
