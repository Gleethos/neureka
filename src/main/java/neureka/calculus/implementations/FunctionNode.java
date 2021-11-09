package neureka.calculus.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.fun.ExecutionDispatcher;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;


public class FunctionNode implements Function
{
    private static Logger _LOG = LoggerFactory.getLogger(FunctionNode.class);

    private final Operation _operation;
    private final boolean _isFlat;
    private final boolean _isDoingAD;

    private final Function[] _src;

    //------------------------------------------------------------------------------------------------------------------

    /**
     *
     * @param type
     * @param sources
     * @param doAD
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

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild( String expression ) {
        return new FunctionBuilder( Neureka.get().context() ).build( expression, true );
    }

    //---

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
        for ( Function f : _src ) if ( f.dependsOn( index ) ) return true;
        return false;
    }

    @Override
    public Function getDerivative( int index ) { return Function.of( _operation.asDerivative( _src, index ) ); }

    @Override
    public List<Function> getSubFunctions() { return Arrays.asList(this._src); }

    //------------------------------------------------------------------------------------------------------------------


    @Override
    public Tsr<?> execute( Args arguments, Tsr<?>... tensors ) {
        return preprocess(
                tensors,
                this,
                () -> {
                    ExecutionCall<? extends Device<?>> call = ExecutionCall.of( tensors )
                                                                            .andArgs(arguments.getAll(Arg.class))
                                                                            .running(_operation)
                                                                            .on( _deviceFor( tensors ) );
                    int d = arguments.valOf(Arg.DerivIdx.class);

                    if ( _isFlat )
                    {
                        /*  The following code is reached in flat functions only:
                            Autograd-Graph will be generated below for the new GraphNode:
                            only flat functions can be executed directly                         */

                        if ( d < 0 && _isDoingAD )
                            return new GraphNode<>(
                                        this,
                                        call,
                                        () -> _execute( call )
                                    ).getPayload();
                    }
                    return _execute( call );
                }
        );
    }


    private Tsr<?> _execute(ExecutionCall<? extends Device<?>> call )
    {
        Tsr<?> alternative = call.getAlgorithm().dispatch( this, call );
        if ( alternative != null ) return alternative;
        throw new IllegalStateException(
                "Missing return value of "+ ExecutionDispatcher.class.getSimpleName() +" in algorithm '"+call.getAlgorithm().getClass().getSimpleName()+"' in operation '"+call.getOperation().getClass().getName()+"'"
        );
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    public Operation getOperation() { return this._operation; }

    public boolean isFlat() { return this._isFlat; }

    public boolean isDoingAD() { return this._isDoingAD; }


    private Tsr<?> preprocess(
            Tsr<?>[] inputs,
            Function function,
            Supplier<Tsr<?>> activation
    ) {
        if ( !function.isDoingAD() ) {
            return activation.get(); // TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES
        }
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
        if ( untracked == null || !allLocked ) { // If graph tracking (nodes) has not yet been initialized!
            return commit( inputs, function, activation );
        }
        GraphLock lock =  untracked.get( GraphNode.class ).getLock();
        attachGraph(inputs, function, lock);
        return activation.get();
    }

    private static Tsr<?> commit(
            Tsr<?>[] inputs, Function function, Supplier<Tsr<?>> activation
    ) {
        Tsr.makeFit( inputs, function.isDoingAD() ); // reshaping if needed

        GraphLock newLock = new GraphLock( function );
        attachGraph( inputs, function, newLock );
        Tsr<?> result;
        if ( activation == null ) result = function.execute( inputs );
        else result = activation.get();

        newLock.release();
        return result;
    }

    private static void attachGraph(
            Tsr<?>[] inputs,
            Function function,
            GraphLock newLock
    ) {
        for ( Tsr<?> t : inputs ) {
            if ( t.has( GraphNode.class ) ) t.get( GraphNode.class ).obtainLocking( newLock );
            else new GraphNode<>( function, newLock, () -> t );
        }
    }


}