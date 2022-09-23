package neureka.calculus.implementations;

import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.Result;
import neureka.backend.main.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.devices.Device;
import neureka.devices.host.CPU;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 *  The most common type of {@link Function} which references other {@link Function}s to
 *  form an abstract syntax tree.
 */
public final class FunctionNode implements Function
{
    private final Operation _operation;
    private final boolean _isFlat;
    private final boolean _isDoingAD;

    private final Function[] _src;


    /**
     * @param type The operation which ought to be represented.
     * @param sources The child function nodes of this node.
     * @param doAD A flag determining if this function should perform autograd.
     */
    public FunctionNode( Operation type, List<Function> sources, boolean doAD )
    {
        if ( type.getArity() >= 0 && sources.size() != type.getArity() ) {
            String tip = ( type.isIndexer() )
                    ? "\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
                    : "";
            throw new IllegalArgumentException(
                    "The function/operation '" + type.getOperator() + "' expects " + type.getArity() + " parameters, " +
                            "however " + sources.size() + " where given!" + tip
            );
        }
        boolean isFlat = true;
        for ( Function f : sources ) // AbstractFunction does only reference tip nodes of the function graph:
            isFlat = (
                    (f instanceof FunctionInput) || (f instanceof FunctionVariable) || (f instanceof FunctionConstant)
            ) && isFlat;

        _operation = type;
        _isFlat = isFlat;
        _src = sources.toArray( new Function[0] );
        _isDoingAD = doAD;
    }

    @Override
    public String toString()
    {
        return _operation.stringify(
                Arrays.stream( _src )
                        .map( e -> e == null ? "(null)" : e.toString() )
                        .toArray( String[]::new )
        );
    }

    @Override
    public boolean dependsOn( int index ) {
        for ( Function f : _src )
            if ( f.dependsOn( index ) ) return true;
        return false;
    }

    @Override
    public Function getDerivative( int index ) { return Function.of( _operation.asDerivative( _src, index ) ); }

    @Override
    public List<Function> getSubFunctions() { return Arrays.asList(_src); }

    @Override
    public Tsr<?> execute( Args arguments, Tsr<?>... inputs ) {
        return _preprocess(
                inputs,
                this,
                () -> {
                    ExecutionCall<? extends Device<?>> call = ExecutionCall.of(inputs)
                                                                            .andArgs( arguments.getAll(Arg.class) )
                                                                            .running(_operation)
                                                                            .on( _deviceFor(inputs) );

                    if ( _isFlat ) call.checkArity();

                    int d = arguments.valOfOr( Arg.DerivIdx.class, -1 );

                    if ( _isFlat )
                    {
                        /*  The following code is reached in flat functions only:
                            Autograd-Graph will be generated below for the new GraphNode:
                            only flat functions can be executed directly                         */

                        if ( d < 0 && _isDoingAD ) {
                            Result[] ref = {null}; // We need to keep a reference so that the garbage collector does not collect the result!
                            new GraphNode<>(
                                    this,
                                    call,
                                    () -> { // This "ref" is a bit of a hack... TODO: fix
                                        ref[0] = _execute(call);
                                        return ref[0];
                                    }
                            );
                            return ref[0].get();
                        }
                    }
                    return _execute( call ).get();
                }
        );
    }

    private Result _execute( ExecutionCall<? extends Device<?>> call )
    {
        return call.getOperation().execute( this, call );
    }

    /**
     *  This method tries to find a common {@link Device} for the provided {@link Tsr}s.
     *
     * @param inputs The input {@link Tsr}s for which a {@link Device} ought to be found and returned.
     * @return A found {@link Device} implementation instance.
     */
    private Device<?> _deviceFor( Tsr<?>[] inputs )
    {
        if ( inputs.length == 0 ) return CPU.get();
        Device<?> device = inputs[ 0 ].get( Device.class );
        boolean onSameDevice = _shareGuestDevice( inputs );
        boolean doAccel = !_operation.getOperator().equals(",") && onSameDevice;
        return ( doAccel && device != null ? device : inputs[ 0 ].getDevice() );
    }

    /**
     * @param tensors An array of tensors for which the most common {@link Device} should be determined.
     * @return The most common {@link Device} among the provided tensors.
     */
    private static boolean _shareGuestDevice( Tsr<?>[] tensors )
    {
        boolean onSameGuestDevice = true;
        Device<?> device = null;
        for ( Tsr<?> tensor : tensors ) device = ( tensor.isOutsourced() ? tensor.get( Device.class ) : device );

        if ( device != null ) {
            for ( Tsr<?> tsr : tensors ) {
                onSameGuestDevice = ( !tsr.isVirtual() && device == tsr.get(Device.class) ) && onSameGuestDevice;
            }
        }
        else onSameGuestDevice = false;

        if ( device != null && tensors.length == 2 && tensors[ 1 ].size() == 1 ) onSameGuestDevice = true;
        return onSameGuestDevice;
    }

    @Override
    public double call( final double[] inputs, int j ) {
        return this.getOperation().calculate( inputs, j, -1, _src );
    }

    @Override
    public double derive( final double[] inputs, final int d, final int j ) {
        return this.getOperation().calculate( inputs, j, d, _src );
    }

    @Override
    public double derive( final double[] inputs, final int d ) {
        return this.getOperation().calculate( inputs, -1, d, _src );
    }

    @Override
    public Operation getOperation() { return _operation; }

    @Override
    public boolean isFlat() { return _isFlat; }

    @Override
    public boolean isDoingAD() { return _isDoingAD; }

    private Tsr<?> _preprocess(
            Tsr<?>[] inputs,
            Function function,
            Supplier<Tsr<?>> activation
    ) {
        if ( !function.isDoingAD() )
            return activation.get(); // TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES

        boolean allLocked = true; // Input tensors might all have graph nodes which are left from previous computation.
        // ( => needs to be locked again! )
        Tsr<?> untracked = null;
        for ( Tsr<?> t : inputs ) {
            GraphNode<?> node = t.get( GraphNode.class );
            if ( node != null ) {
                untracked = t;
                allLocked = node.getLock().isLocked() && allLocked;
            }
        }
        if ( untracked == null || !allLocked )  // If graph tracking (nodes) has not yet been initialized!
            return _commit( inputs, function, activation );

        GraphLock lock =  untracked.get( GraphNode.class ).getLock();
        _attachGraph( inputs, function, lock );
        return activation.get();
    }

    private static Tsr<?> _commit(
            Tsr<?>[] inputs,
            Function function,
            Supplier<Tsr<?>> activation
    ) {
        Reshape.makeFit( inputs, function.isDoingAD() ); // reshaping if needed

        GraphLock newLock = new GraphLock( function );
        _attachGraph( inputs, function, newLock );
        Tsr<?> result;
        if ( activation == null ) result = function.execute( inputs );
        else result = activation.get();

        newLock.release();
        return result;
    }

    private static void _attachGraph(
            Tsr<?>[] inputs,
            Function function,
            GraphLock newLock
    ) {
        for ( Tsr<?> t : inputs ) {
            if ( t.has( GraphNode.class ) ) t.get( GraphNode.class ).obtainLocking( newLock );
            else new GraphNode<>( function, newLock, () -> Result.of(t) );
        }
    }


}