
package neureka.backend.api.operations;


import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.FallbackAlgorithm;
import neureka.backend.standard.algorithms.Activation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.calculus.implementations.FunctionConstant;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 *  This abstract {@link Operation} implementation is a useful template for creating new operations.
 *  It provides a partial implementation which consists of a simple component system for hosting {@link Algorithm} instances
 *  as well as a set of properties which {@link Operation} implementations are expected to have. <br>
 *  Therefore, the number of properties this class needs to receive is rather large.
 *  In order to instantiate it one has to pass {@link OperationBuilder} instance to the constructor.
 *  Using the factory will make the property configuration as readable as possible. <br>
 *
 */
public abstract class AbstractOperation implements Operation
{
    private static Logger _LOG = LoggerFactory.getLogger( AbstractOperation.class );

    /**
     *  An operation may have two ways in which it can describe itself as String within a Function AST.
     *  The first one is an operator style of representation and the second one a classical function.
     *  So for the 'Addition' operation the following two representations exist: <br>
     * <ul>
     *      <li> Operator: '+';   Example: 'I[0] + 3 + 5 * I[1]'
     *      <li> Function: 'add'; Example: 'add( I[0], 3, 5*I[1] )'
     * </ul>
     * The following String is the latter way of representing the operation, namely: a functional way.
     */
    protected final String _function;

    /**
     *  An operation may have two ways in which it can describe itself as String within a Function AST.
     *  The first one is an operator style of representation and the second one a classical function.
     *  So for the 'Addition' operation the following two representations exist: <br>
     * <ul>
     *      <li> Operator: '+';   Example: 'I[0] + 3 + 5 * I[1]'
     *      <li> Function: 'add'; Example: 'add( I[0], 3, 5*I[1] )'
     * </ul>
     * The following String is the primer way of representing the operation, namely: as an operator.
     */
    protected final String _operator;

    /**
     * Arity is the number of arguments or operands
     * that this function or operation takes.
     */
    protected final int _arity;

    /**
     *  This flag determines if this operation is auto-indexing passed input arguments.
     *  Auto-indexing inputs means that for a given array of input arguments
     *  the wrapping Function instance will call its child nodes targeted via an
     *  index incrementally.
     *  The variable 'j' in a Functions expressions containing 'I[j]' will then be
     *  resolved to an actual input for a given indexer...
     */
    protected final boolean _isIndexer;

    /**
     *  Certain operations are not differentiable, meaning they cannot participate
     *  in neither forward- or reverse- mode differentiation.
     *  In order to avoid error prone behaviour trying involve
     *  non- differentiable operations will yield proper exceptions.
     */
    protected final boolean _isDifferentiable;

    /**
     *  Inline operations are operations which change the state of the arguments passed to them.
     *
     */
    protected final boolean _isInline;
    protected final boolean _isOperator;

    private final Map<Class<?>, Algorithm<?>> _algorithms = new LinkedHashMap<>();

    /**
     *  This is the default algorithm for every Operation extending this class.
     *  It may not fit the purpose of every Operation implementation,
     *  however for most operation types it will provide useful functionalities.
     *
     *  The default algorithm assumes an operation that is either a function or operator.
     *  Meaning that it assumes that the operation is also differentiable.
     *  Therefore it contains functionality that goes alongside this assumption,
     *  just to name a few :                                                                                            <br>
     *                                                                                                                  <br>
     *  - An ADAgent supplier returning ADAgent instances capable of performing both forward- and reverse- mode AD.     <br>
     *  - A simple result tensor instantiation implementation.                                                          <br>
     *  - A basic threaded execution based on the AST of a given Function object.                                       <br>
     */
    private final Algorithm<?> _defaultAlgorithm;

    public AbstractOperation( OperationBuilder builder )
    {
        builder.dispose();

        _function         = builder.getFunction();
        _arity            = builder.getArity();
        _operator         = builder.getOperator();
        _isOperator       = builder.getIsOperator();
        _isIndexer        = builder.getIsIndexer();
        _isDifferentiable = builder.getIsDifferentiable();
        _isInline         = builder.getIsInline();
        _defaultAlgorithm = new FallbackAlgorithm( "default", _arity, this );
    }

    public static Tsr<?> executeMe(
        Function caller,
        ExecutionCall<? extends Device<?>> call
    ) {
        Function[] nodes = caller.getSubFunctions().toArray(new Function[0]);
        Operation operation = caller.getOperation();
        boolean isFlat = caller.isFlat();
        boolean isDoingAD = caller.isDoingAD();
        if ( call.getDerivativeIndex() < 0 ) return _deepActivation( call, nodes, operation, isFlat, isDoingAD );
        else return _deepDerivative( call, nodes, operation );
    }

    private static Tsr<?> _deepActivation(
            ExecutionCall<? extends Device<?>> call,
            Function[] nodes,
            Operation operation,
            boolean isFlat,
            boolean isDoingAD
    ) {
        Tsr<?>[] inputs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getDerivativeIndex();
        int j = call.getJ();

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
            tensors = srcActivation(inputs, j, d, 0, nodes);
            String asStr = operation.stringify(
                    IntStream.range(0, nodes.length).mapToObj(i -> "I[" + i + "]").toArray(String[]::new)
            );
            return new FunctionBuilder(Neureka.get().context()).build( asStr, isDoingAD ).execute( tensors );
        } else {
            tensors = srcActivation(inputs, j, d, 1, nodes);
        }
        device.execute(
                ExecutionCall.of(tensors)
                        .andArgs(Arg.DerivIdx.of(d))
                        .running(operation)
                        .on(device)
        );
        if ( tensors[ 0 ] == null ) _LOG.warn("Function did not have a proper return value.");
        return ( tensors[ 0 ] == null ) ? tensors[ 1 ] : tensors[ 0 ];
    }

    /**
     *  This method return the index of the tensor
     *  in the given tensor array which is virtual and contains "1.0".
     *  However if not all tensors are virtual or their values are not all "0.0" except one
     *  whose value is "1.0" then it return -1, because the optimization cannot
     *  be made...
     *
     * @param tensors An array of tensors which ought to be analyzed.
     * @return The index of the tensor whose value is "1.0" (if all other are "0.0"), otherwise : -1
     */
    private static int _indexOfFoundDerivative(Tsr<?>[] tensors )
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

    private static Tsr<?> _deepDerivative(
            ExecutionCall<? extends Device<?>> call,
            Function[] nodes,
            Operation operation
    ) {
        Supplier<Tsr<?>> actor = () -> {
            Tsr<?>[] inputs = call.getTensors();
            Device<?> device = call.getDevice();
            int d = call.getDerivativeIndex();
            int j = call.getJ();

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
                    device.execute(
                            ExecutionCall.of(tensors)
                                    .andArgs(Arg.DerivIdx.of( -1 ))
                                    .running(Neureka.get().context().getOperation("+"))
                                    .on(device)
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
                if ( nodes[ i ].dependsOn(d) && !operation.isIndexer() ) {
                    d = i;
                    break;
                }
            }
            // Use those tensors for the outer derivative:
            device.execute(
                    ExecutionCall.of(tensors)
                            .andArgs(Arg.DerivIdx.of(d))
                            .running( operation )
                            .on( device )
            );
            // At the end:
            //...multiply inner times outer: ( if inner is not 1 entirely... )
            if ( !( ( inner.isVirtual() || inner.size()==1 ) && inner.value64( 0 )==1.0) ) {
                tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
                device.execute(
                        ExecutionCall.of(tensors)
                                .andArgs(Arg.DerivIdx.of( -1 ))
                                .running(Neureka.get().context().getOperation("*"))
                                .on( device )
                );
            } // done!
            return tensors[ 0 ];
        };

        Device<?> device = call.getDevice();
        int d = call.getDerivativeIndex();
        Tsr<?> out = null;
        for ( int i = 0; i < nodes.length; i++ ) { // constants need to be figured out!
            int di = ( nodes[ i ].dependsOn(d) ) ? i : -1;
            if ( di >= 0 ) {
                if ( out == null ) out = actor.get();
                else
                    device.execute(
                            ExecutionCall.of( null, actor.get(), out )
                                    .andArgs( Arg.DerivIdx.of( -1 ) )
                                    .running( Neureka.get().context().getOperation("+") )
                                    .on( device )
                    );
            }
        }
        return out;
    }

    public static Tsr<?>[] srcActivation(
            Tsr<?>[] inputs, int j, int d, int offset, Function[] src
    ) {
        int[] tempShape = null;
        Tsr<?>[] tensors = new Tsr[ src.length + offset ];
        for ( int i = offset; i < tensors.length; i++ ) {//constants need to be figured out!
            if ( !(src[ i - offset ] instanceof FunctionConstant) ) {
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

                tempShape = ( tempShape == null ) ? tensors[ i ].getNDConf().shape() : tempShape;
            }
        }
        for ( int i = offset; i < tensors.length; i++ ) {
            if ( tensors[ i ] == null )
                tensors[ i ] =
                        ( j < 0 )
                                ? Tsr.of(tempShape, ((FunctionConstant) src[ i - offset ]).value())
                                : Tsr.of(tempShape, src[ i - offset ].call(new double[]{}, j));
        }
        return tensors;
    }

    //==================================================================================================================

    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method ensures this in terms of read access by returning only a single instance or null
     *  based on the provided class instance whose type extends the {@link Algorithm} interface.
     *
     * @param type The class of the type which implements {@link Algorithm} as a key to get an existing instance.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The instance of the specified type if any exists within this {@link Operation}.
     */
    @Override
    public <T extends Algorithm<T>> T getAlgorithm( Class<T> type ) {
        T found = (T) _algorithms.get( type );
        if ( found == null ) { // Maybe the provided type is a superclass of one of the entries...
            return _algorithms.entrySet()
                                    .stream()
                                    .filter( e -> type.isAssignableFrom( e.getKey() ) )
                                    .map( e -> (T) e.getValue() )
                                    .findFirst()
                                    .orElse( null );
        }
        else
            return found;
    }

    /**
     *  This method checks if this {@link Operation} contains an instance of the
     *  {@link Algorithm} implementation specified via its type class.
     *
     * @param type The class of the type which implements {@link Algorithm}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return The truth value determining if this {@link Operation} contains an instance of the specified {@link Algorithm} type.
     */
    @Override
    public <T extends Algorithm<T>> boolean supportsAlgorithm( Class<T> type ) {
        return _algorithms.containsKey( type );
    }

    /**
     *  {@link Operation} implementations embody a component system hosting unique {@link Algorithm} instances.
     *  For a given class implementing the {@link Algorithm} class, there can only be a single
     *  instance of it referenced (aka supported) by a given {@link Operation} instance.
     *  This method enables the registration of {@link Algorithm} types in the component system of this {@link Operation}.
     *
     * @param type The class of the type which implements {@link Algorithm} as key for the provided instance.
     * @param instance The instance of the provided type class which ought to be referenced (supported) by this {@link Operation}.
     * @param <T> The type parameter of the {@link Algorithm} type class.
     * @return This very {@link Operation} instance to enable method chaining on it.
     */
    @Override
    public <T extends Algorithm<T>> Operation setAlgorithm( Class<T> type, T instance ) {
        _algorithms.put( type, instance );
        return this;
    }

    //==================================================================================================================

    @Override
    public <T extends Algorithm<T>> Algorithm<T> getAlgorithmFor( ExecutionCall<?> call ) {
        float bestScore = 0f;
        Algorithm<T> bestImpl = null;
        for( Algorithm<?> impl : _algorithms.values() ) {
            float currentScore = impl.isSuitableFor( call );
            if ( currentScore > bestScore ) {
                if ( currentScore == 1.0 ) return (Algorithm<T>) impl;
                else {
                    bestScore = currentScore;
                    bestImpl = (Algorithm<T>) impl;
                }
            }
        }

        if ( _defaultAlgorithm.isSuitableFor( call ) > 0.0f ) return (Algorithm<T>) _defaultAlgorithm;

        if ( bestImpl == null ) {
            String message = "No suitable implementation for execution call '"+call+"' could be found.\n" +
                    "Execution process aborted.";
            _LOG.error( message );
            throw new IllegalStateException( message );
        }
        return bestImpl;
    }

    //==================================================================================================================

    @Override
    public <T extends Algorithm<T>> boolean supports( Class<T> implementation ) {
        return _algorithms.containsKey( implementation );
    }

    @Override
    public boolean isOperator() {
        return _isOperator;
    }


    public String getFunction() {
        return this._function;
    }

    public String getOperator() {
        return this._operator;
    }

    public int getArity() {
        return this._arity;
    }

    public boolean isIndexer() {
        return this._isIndexer;
    }

    public boolean isDifferentiable() {
        return this._isDifferentiable;
    }

    public boolean isInline() {
        return this._isInline;
    }

    public Algorithm<?> getDefaultAlgorithm() {
        return this._defaultAlgorithm;
    }
}
