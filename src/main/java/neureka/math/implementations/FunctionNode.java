package neureka.math.implementations;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.main.operations.other.Permute;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.args.Args;

import java.util.Arrays;
import java.util.List;

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
        for ( int i = 0; i < _src.length; i++ ) {
            if ( _src[i] == null )
                throw new IllegalArgumentException("The function node '" + this + "' has a null source at index " + i + "!");
            if ( _src[i] instanceof FunctionNode && _src[i].isDoingAD() != _isDoingAD )
                throw new IllegalArgumentException(
                        "Detected an attempt to mix autograd and non-autograd functions in the same function graph!\n" +
                        "A function can either be doing autograd or not doing autograd!"
                    );
        }
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
    public Tensor<?> execute(Args arguments, Tensor<?>... inputs )
    {
        if ( this.isDoingAD() )
            Permute.makeFit( inputs, this.isDoingAD() ); // reshaping if needed

        ExecutionCall<? extends Device<?>> call = ExecutionCall.of(inputs)
                                                                .andArgs(arguments.getAll(Arg.class))
                                                                .running(_operation)
                                                                .on(_deviceFor(inputs));
        return call.getOperation()
                .execute( this, call ).get();
    }

    /**
     *  This method tries to find a common {@link Device} for the provided {@link Tensor}s.
     *
     * @param inputs The input {@link Tensor}s for which a {@link Device} ought to be found and returned.
     * @return A found {@link Device} implementation instance.
     */
    private Device<?> _deviceFor( Tensor<?>[] inputs )
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
    private static boolean _shareGuestDevice( Tensor<?>[] tensors )
    {
        boolean onSameGuestDevice = true;
        Device<?> device = null;
        for ( Tensor<?> tensor : tensors ) device = ( tensor.isOutsourced() ? tensor.get( Device.class ) : device );

        if ( device != null ) {
            for ( Tensor<?> tensor : tensors ) {
                onSameGuestDevice = ( !tensor.isVirtual() && device == tensor.get(Device.class) ) && onSameGuestDevice;
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

}