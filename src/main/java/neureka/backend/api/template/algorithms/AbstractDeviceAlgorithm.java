package neureka.backend.api.template.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.*;
import neureka.backend.api.fun.ExecutionPreparation;
import neureka.backend.main.algorithms.ElementwiseAlgorithm;
import neureka.backend.main.internal.FinalExecutor;
import neureka.backend.main.memory.MemUtil;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;
import neureka.math.implementations.FunctionConstant;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  This is a partial implementation of the {@link Algorithm} interface which implements
 *  the component system for implementation instances of the {@link ImplementationFor} interface.
 *  These components implement an algorithm for a specific {@link Algorithm}.
 *
 * @param <C> The type of the concrete extension of this class.
 */
public abstract class AbstractDeviceAlgorithm<C extends DeviceAlgorithm<C>>
extends AbstractAlgorithm
implements DeviceAlgorithm<C>
{
    private final static Logger _LOG = LoggerFactory.getLogger(AbstractDeviceAlgorithm.class);

    protected final Map<Class<Device<?>>, ImplementationFor<Device<?>>> _implementations = new HashMap<>();

    public AbstractDeviceAlgorithm( String name ) { super( name ); }

    @Override
    public <D extends Device<?>, E extends ImplementationFor<D>> C setImplementationFor(
            Class<D> deviceClass, E implementation
    ) {
        if ( _implementations.containsKey( deviceClass ) )
            _LOG.info(
                "Implementation for device '" + deviceClass.getSimpleName() + "' already defined!"
            );

        _implementations.put(
            (Class<Device<?>>) deviceClass,
            (ImplementationFor<Device<?>>) implementation
        );
        return (C) this;
    }

    @Override
    public <D extends Device<?>> ImplementationFor<D> getImplementationFor(Class<D> deviceClass )
    {
        ImplementationFor<D> found = (ImplementationFor<D>) _implementations.get( deviceClass );
        if ( found == null )
            for ( Class<Device<?>> type : _implementations.keySet() )
                if ( type.isAssignableFrom(deviceClass) )
                    return (ImplementationFor<D>) _implementations.get(type);

        return found;
    }

    @Override
    public String toString() {
        String algorithmString = getClass().getSimpleName()+"@"+Integer.toHexString(hashCode());
        String implementations = _implementations.keySet().stream().map(Class::getSimpleName).collect(Collectors.joining(","));
        algorithmString = ( algorithmString + "[name=" + getName() + ",support=[" + implementations + "]]" );
        return algorithmString;
    }


    public static Tsr<?> executeFor(
            final Function caller,
            final ExecutionCall<? extends Device<?>> call,
            final FinalExecutor executor
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

    public static Tsr<?> prepareAndExecute(
            ExecutionCall<? extends Device<?>> executionCall,
            FinalExecutor executor
    ) {
        ExecutionCall<? extends Device<?>> call = _prepareForExecution(executionCall);
        return executeOnCommonDevice(call, ()->{
             /*
                Below is the core lambda of recursive preprocessing
                which is defined for each Algorithm individually :
             */
            Tsr<?> result = null;
            if ( executor != null )
                result = executor.execute(call);
            return result;
        });
    }

    public static ExecutionCall<? extends Device<?>> _prepareForExecution(ExecutionCall<? extends Device<?>> executionCall) {
        Algorithm currentAlgorithm = executionCall.getAlgorithm();
        if ( currentAlgorithm instanceof ExecutionPreparation)
            executionCall = ( (ExecutionPreparation) currentAlgorithm ).prepare( executionCall );

        for ( Tsr<?> t : executionCall.inputs() )
            if ( t == null ) throw new IllegalArgumentException(
                                "Device arguments may not be null!\n" +
                                "One or more tensor arguments within the given ExecutionCall instance is null."
                            );
        return executionCall;
    }

    public static Tsr<?> executeDeviceAlgorithm(
            ExecutionCall<? extends Device<?>> call
    ) {
        for ( Tsr<?> t : call.inputs() )
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                    "One or more tensor arguments within the given ExecutionCall instance is null."
                );

        Device<?> device = call.getDevice();

        Algorithm algorithm = call.getAlgorithm();
        if ( algorithm == null ) {
            String message = _couldNotFindSuitableAlgorithmFor( device.getClass() );
            _LOG.error( message );
            throw new IllegalStateException( message );
        } else {
            DeviceAlgorithm<?> deviceAlgorithm = ( algorithm instanceof DeviceAlgorithm ? ((DeviceAlgorithm<?>) algorithm) : null );
            ImplementationFor<Device<?>> implementation =  ( deviceAlgorithm == null ? null : deviceAlgorithm.getImplementationFor(device) );
            if ( implementation == null ) {
                String message = _couldNotFindSuitableImplementationFor( call.getOperation(), algorithm, device.getClass() );
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
            else {
                device.approve( call );
                return implementation.run( (ExecutionCall<Device<?>>) call );
            }
        }
    }

    public static <D extends Device<?>> ExecutionCall<D> flatten(
            Function caller, ExecutionCall<D> call
    ) {
        return _flatten( call, caller.getSubFunctions().toArray(new Function[0]), true );
    }

    public static <D extends Device<?>> ExecutionCall<D> flattenForIndexer(
            Function caller, ExecutionCall<D> call
    ) {
        return _flatten( call, caller.getSubFunctions().toArray(new Function[0]), false );
    }

    private static <D extends Device<?>> ExecutionCall<D> _flatten(
            ExecutionCall<D> call, Function[] src
    ) {
        return _flatten( call, src, true );
    }

    
    private static <D extends Device<?>> ExecutionCall<D> _flatten(
            ExecutionCall<D> call, Function[] src, boolean ignoreJs
    ) {
        ExecutionCall<D> innerCall = !ignoreJs ? call : call.withArgs( Arg.DerivIdx.of(-1) );
        Tsr<?>[] inputs = innerCall.inputs();
        return MemUtil.keep( inputs, () ->
        {
            int[] tempShape = null;
            Class<?> tempType = null;
            Tsr<?>[] tensors = new Tsr[src.length];
            for ( int i = 0; i < tensors.length; i++ ) {//constants need to be figured out!
                if ( !( src[i] instanceof FunctionConstant) ) {
                    tensors[ i ] = src[i].execute(innerCall);
                    tempShape = ( tempShape == null ? tensors[ i ].getNDConf().shape() : tempShape );
                    tempType  = ( tempType  == null ? tensors[ i ].getItemType()     : tempType  );
                }
            }
            int j = innerCall.getValOf( Arg.VarIdx.class );
            for ( int i = 0; i < tensors.length; i++ )
                if ( tensors[ i ] == null )
                    tensors[ i ] =
                            j < 0
                                ? Tsr.of( tempType, tempShape, ((FunctionConstant) src[i]).value() ).mut().setIsIntermediate( true ).to(call.getDevice())
                                : Tsr.of( tempType, tempShape, src[i].call(new double[]{}, j)      ).mut().setIsIntermediate( true ).to(call.getDevice());

            return innerCall.withInputs(tensors);
        });
    }

    
    private static Tsr<?> _deepActivation(
            final ExecutionCall<? extends Device<?>> call,
            final Function[] nodes,
            final boolean isFlat,
            final boolean isDoingAD,
            final FinalExecutor executor
    ) {
        int j = call.getValOf( Arg.VarIdx.class );
        assert call.getValOf( Arg.DerivIdx.class ) == -1;

        ExecutionCall<?> flattenedCall = _flatten( call.withArgs( Arg.VarIdx.of(j) ), nodes );

        if (
                !isFlat && j < 0 && (
                        call.getOperation().isOperator()
                                ||
                        call.getOperation().supportsAlgorithm(ElementwiseAlgorithm.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            String asStr = call.getOperation().stringify(
                                        IntStream.range(0, nodes.length)
                                            .mapToObj(i -> "I[" + i + "]")
                                            .toArray(String[]::new)
                                    );
            Tsr<?>[] finalTensors = flattenedCall.inputs();
            Tsr<?> result = MemUtil.keep(finalTensors, () -> new FunctionParser(Neureka.get().backend()).parse(asStr, isDoingAD).execute(finalTensors));
            for ( int i = 1; i < finalTensors.length; i++ )
                _deleteIfNotIn(call.inputs(), finalTensors[i]);

            return result;
        } else {
            int numberOfInputs = flattenedCall.arity();
            boolean anyNumberOfInputs = flattenedCall.getOperation().getArity() < 0;
            int operationArity = flattenedCall.getOperation().getArity();
            if (numberOfInputs < operationArity)
                throw new IllegalArgumentException(
                        "The number of inputs to the operation " + flattenedCall.getOperation() + " is " + numberOfInputs +
                        " but the operation requires " + operationArity + " inputs."
                    );

            boolean tooManyArgs = numberOfInputs > operationArity + 1;

            Tsr<?>[] tensors;

            if ( !tooManyArgs || anyNumberOfInputs )
                tensors = flattenedCall.withAddedInputAt(0, null).inputs();
            else
                tensors = flattenedCall.inputs();

            return prepareAndExecute(
                        call.withInputs( tensors ).withArgs( Arg.DerivIdx.of(-1), Arg.VarIdx.of(-1) ),
                        executor
                    );
        }
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

    
    private static Tsr<?> _deepDerivative(
            final ExecutionCall<? extends Device<?>> call,
            final Function[] nodes,
            final FinalExecutor executor
    ) {
        Supplier<Tsr<?>> actor = () ->
                MemUtil.keep( call.inputs(), () -> {
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
                            tensors[0] = prepareAndExecute(
                                                ExecutionCall.of( tensors )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("+") )
                                                        .on( call.getDevice() ),
                                                innerCall -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( innerCall )
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
                    tensors[0] = prepareAndExecute(
                                        ExecutionCall.of( tensors )
                                                .andArgs( Arg.DerivIdx.of( d ) )
                                                .running( call.getOperation() )
                                                .on( call.getDevice() ),
                                        executor
                                    );
                    // At the end:
                    //...multiply inner times outer: ( if inner is not 1 entirely... )
                    Tsr<?> result = _innerTimesOuter( inner, tensors, call );
                    // done!

                    _delete( inner );

                    return result;
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

    private static Tsr<?> _innerTimesOuter(Tsr<?> inner, Tsr<?>[] tensors, ExecutionCall<?> call)
    {
        if ( !( ( inner.isVirtual() || inner.size() == 1 ) && inner.getItemsAs( double[].class )[ 0 ] == 1.0 ) ) {
            tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
            tensors[0] = prepareAndExecute(
                    ExecutionCall.of( tensors )
                            .andArgs( Arg.DerivIdx.of( -1 ) )
                            .running( Neureka.get().backend().getOperation("*") )
                            .on( call.getDevice() ),
                    AbstractDeviceAlgorithm::executeDeviceAlgorithm
            );
            for ( int i = 1; i < tensors.length; i++ )
                _deleteIfNotIn( call.inputs(), tensors[ i ] );
        }
        return tensors[ 0 ];
    }

    private static void _deleteIfNotIn( Tsr<?>[] array, Tsr<?> tensor ) {
        if ( Neureka.get().settings().debug().isDeletingIntermediateTensors() ) {
            for ( int i = 1; i < array.length; i++ )
                if ( array[i] == tensor ) return;

            if ( !tensor.isDeleted() ) tensor.mut().delete();
        }
    }

    private static void _delete( Tsr<?> tensor ) {
        Neureka.Settings.Debug debug = Neureka.get().settings().debug();
        if (  !tensor.isDeleted() && debug.isDeletingIntermediateTensors() )
            tensor.mut().delete();
    }

    public static <R> R executeOnCommonDevice( ExecutionCall<?> call, Supplier<R> execution ) {
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

        for ( int i = 0; i < rollbacks.length; i++ )
            if ( call.input( i ) != null && !call.input( i ).isDeleted() && !call.input( i ).isUndefined() )
                rollbacks[ i ].accept( call.input( i ) );

        return result;
    }

    private static String _couldNotFindSuitableAlgorithmFor( Class<?> type ) {
        return LogUtil.format(
                "No suitable '"+ Algorithm.class.getSimpleName()+"' found for device of type '{}'.",
                type.getSimpleName()
        );
    }

    private static String _couldNotFindSuitableImplementationFor(
            Operation operation,
            Algorithm algorithm,
            Class<?> type
    ) {
        return LogUtil.format(
                "No suitable implementation found for operation '{}', algorithm '{}' and device type '{}'.",
                operation.getIdentifier(),
                algorithm.getName(),
                type.getSimpleName()
        );
    }


}
