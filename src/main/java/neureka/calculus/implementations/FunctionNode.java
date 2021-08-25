package neureka.calculus.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.standard.algorithms.Activation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


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
        _src = sources.toArray(new Function[0]);
        _isDoingAD = doAD;
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild( String expression ) {
        return new FunctionBuilder(Neureka.get().context()).build( expression, true );
    }

    //---

    @Override
    public String toString()
    {
        return _operation.stringify(
                Arrays.stream(_src)
                        .map( e -> ( e == null ) ? "(null)" : e.toString() )
                        .collect(Collectors.toList())
                        .toArray(new String[0])
        );
    }

    @Override
    public boolean dependsOn( int index ) {
        for ( Function f : _src ) if ( f.dependsOn(index) ) return true;
        return false;
    }

    @Override
    public Function getDerivative( int index ) {
        return Function.of( _operation.asDerivative( _src, index ) );
    }

    @Override
    public List<Function> getSubFunctions() { return Arrays.asList(this._src); }

    //------------------------------------------------------------------------------------------------------------------

    /**
     * Responsible for handling functions with multiple inputs!
     *
     * @param inputs
     * @return The result of the execution.
     */
    protected Tsr _tensor_activation( Tsr[] inputs, Args arguments )
    {
        ExecutionCall<? extends Device<?>> call = ExecutionCall.builder()
                                                                .device(_deviceFor( inputs ))
                                                                .tensors( inputs )
                                                                .operation( _operation )
                                                                .derivativeIndex( arguments.getValOf(Arg.DerivIdx.class) )
                                                                .args(
                                                                    arguments.get(Arg.VarIdx.class)
                                                                )
                                                                .build();
        ExecutionCall<? extends Device<?>> finalCall;
        Device<?> possiblyNewDevice = call.getAlgorithm().findDeviceFor( call );
        if ( possiblyNewDevice != null ) finalCall = call.withDevice( possiblyNewDevice );
        else finalCall = call;

        int d = arguments.getValOf(Arg.DerivIdx.class);

        if ( _isFlat )
        {
            /* The following code is reached in flat functions only: 
               Autograd-Graph will be generated below for the new GraphNode: 
               only flat functions can be executed directly */
            if ( d < 0 && _isDoingAD )
                return new GraphNode<>( this, finalCall, () -> __flat_execution( finalCall ) ).getPayload();
            else
                return __flat_execution( finalCall );
        }/* The code below deals with deep functions (non flat) :  */
        else if ( d < 0 ) return __deep_activation( finalCall );
        else return _deep_derivative( finalCall );

    }

    private Tsr __flat_execution( ExecutionCall<? extends Device<?>> call )
    {
        Tsr alternative = call.getAlgorithm().handleInsteadOfDevice( this, call );
        if ( alternative != null ) return alternative;

        if ( call.getDerivativeIndex() < 0 ) return __deep_activation( call );
        else return _deep_derivative( call  );
    }
 
    private Tsr __deep_activation( ExecutionCall<? extends Device<?>> call )
    {
        Tsr[] inputs = call.getTensors();
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        int j = call.getJ();

        Tsr[] tensors;
        if ( _operation.isIndexer() ) tensors = new Tsr[ 1 + inputs.length ];
        else tensors = new Tsr[ 1 + _src.length ];

        if ( _operation.isIndexer() ) {
            for ( int i = 1; i < tensors.length; i++ ) tensors[ i ] = _src[ 0 ].call( inputs, i - 1 );
        } else if (
                !_isFlat && j < 0 && (
                        _operation.isOperator() || _operation.supportsAlgorithm(Activation.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            tensors = srcActivation(inputs, j, d, 0);
            String asStr = _operation.stringify(
                    IntStream.range(0, _src.length).mapToObj(i -> "I[" + i + "]").toArray(String[]::new)
            );
            return new FunctionBuilder(Neureka.get().context()).build( asStr, _isDoingAD ).call( tensors );
        } else {
            tensors = srcActivation(inputs, j, d, 1);
        }
        device.execute(
                ExecutionCall.builder()
                        .device( device )
                        .tensors( tensors )
                        .derivativeIndex( d )
                        .operation( _operation )
                        //.args(call.findAll(Arg.class))
                        .build()
        );
        if ( tensors[ 0 ] == null ) _LOG.warn("Function '"+this+"' did not have a proper return value.");
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
    private int ___indexOfFoundDerivative( Tsr<?>[] tensors )
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

    private Tsr _deep_derivative( ExecutionCall<? extends Device<?>> call )
    {
        Supplier<Tsr<?>> actor = () -> {
                    Tsr[] inputs = call.getTensors();
                    Device device = call.getDevice();
                    int d = call.getDerivativeIndex();
                    int j = call.getJ();

                    Tsr[] tensors;
                    if ( _operation.isIndexer() ) tensors = new Tsr[ 1 + inputs.length ];
                    else tensors = new Tsr[ 1 + _src.length ];

                    // Chain-rule (forward AutoDiff):
                    // inner times outer means:
                    // first derive source!
                    // like so:
                    if ( _operation.isIndexer() ) {
                        for ( int i = 1; i < tensors.length; i++ ) {
                            tensors[ i ] = _src[ 0 ].derive( inputs, d, i - 1 );
                        }
                    } else {
                        for ( int i = 1; i < tensors.length; i++ ) {
                            tensors[ i ] =
                                    ( j >= 0 )
                                            ? _src[ i - 1 ].derive( inputs, d, j )
                                            : _src[ i - 1 ].derive( inputs, d );
                        }
                    }
                    //...then add them all together! (is possible because of linearity...)
                    Tsr inner;
                    if ( tensors.length > 2 ) {// Optimization: Finds index of "1.0" among otherwise all "0.0" virtual tensors!
                        int index = ___indexOfFoundDerivative( tensors );
                        if ( index >= 0 ) inner = tensors[ index ];
                        else {
                            // Optimization above did not apply, so we accumulate all the derivatives!
                            device.execute(
                                    ExecutionCall.builder()
                                        .device( device )
                                        .tensors( tensors )
                                        .args( Arg.DerivIdx.of( -1 ) )
                                        .operation( Neureka.get().context().instance("+") )
                                        .build()
                            );
                            inner = tensors[ 0 ];//-> this is now the inner derivative!
                        }
                    }
                    else inner = tensors[ 1 ];

                    tensors[ 0 ] = null;
                    //...then activate (No differentiation!) the source like so:
                    if ( _operation.isIndexer() ) { // Indexer pass an index j of course!
                        for ( int i = 1; i < tensors.length; i++ ) {
                            tensors[ i ] = _src[ 0 ].call( inputs, i - 1 ); // i - 1 := j
                        }
                    } else {
                        for ( int i = 1; i < tensors.length; i++ ) {
                            tensors[ i ] = ( j >= 0 ) ? _src[ i - 1 ].call( inputs, j ) : _src[ i - 1 ].call( inputs );
                        }
                    }
                    //...get derivative index within src list:
                    for ( int i = 0; i < _src.length; i++ ) {
                        if ( _src[ i ].dependsOn(d) && !_operation.isIndexer() ) {
                            d = i;
                            break;
                        }
                    }
                    // Use those tensors for the outer derivative:
                    device.execute(
                            ExecutionCall.builder()
                                            .device( device )
                                            .tensors( tensors )
                                            .derivativeIndex( d )
                                            .operation( _operation )
                                            .build()
                    );
                    // At the end:
                    //...multiply inner times outer: ( if inner is not 1 entirely... )
                    if ( !( ( inner.isVirtual() || inner.size()==1 ) && inner.value64( 0 )==1.0) ) {
                        tensors = new Tsr[]{ null, inner, tensors[ 0 ] };
                        device.execute(
                                ExecutionCall.builder()
                                                .device( device )
                                                .tensors( tensors )
                                                .args( Arg.DerivIdx.of( -1 ) )
                                                .operation( Neureka.get().context().instance("*") )
                                                .build()
                        );
                    } // done!
                    return tensors[ 0 ];

                };
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        Tsr out = null;
        for ( int i = 0; i < _src.length; i++ ) { // constants need to be figured out!
            int di = ( _src[ i ].dependsOn(d) ) ? i : -1;
            if ( di >= 0 ) {
                if ( out == null ) out = actor.get();
                else
                    device.execute(
                            ExecutionCall.builder()
                                .device( device )
                                .tensors( new Tsr[]{ null, actor.get(), out } )
                                .args( Arg.DerivIdx.of( -1 ) )
                                .operation( Neureka.get().context().instance("+") )
                                .build()
                );
            }
        }
        return out;
    }

    public Tsr<?>[] srcActivation(
            Tsr<?>[] inputs, int j, int d, int offset
    ) {
        int[] tempShape = null;
        Tsr<?>[] tensors = new Tsr[ _src.length + offset ];
        for ( int i = offset; i < tensors.length; i++ ) {//constants need to be figured out!
            if ( !(_src[ i - offset ] instanceof FunctionConstant) ) {
                if ( d < 0 ) // Not deriving this!
                    tensors[ i ] =
                            ( j >= 0 )
                                    ? _src[ i - offset ].execute( inputs, j )
                                    : _src[ i - offset ].execute( inputs );
                else // ...deriving at specified index...
                    tensors[ i ] =
                            ( j >= 0 )
                                    ? _src[ i - offset ].executeDerive( inputs, d, j )
                                    : _src[ i - offset ].executeDerive( inputs, d );

                tempShape = ( tempShape == null ) ? tensors[ i ].getNDConf().shape() : tempShape;
            }
        }
        for ( int i = offset; i < tensors.length; i++ ) {
            if ( tensors[ i ] == null )
                    tensors[ i ] =
                        ( j < 0 )
                                ? Tsr.of(tempShape, ((FunctionConstant) _src[ i - offset ]).value())
                                : Tsr.of(tempShape, _src[ i - offset ].call(new double[]{}, j));
        }
        return tensors;
    }

    private Device<?> _deviceFor( Tsr<Object>[] inputs )
    {
        if ( inputs.length == 0 ) return HostCPU.instance();
        Device<?> device = inputs[ 0 ].get( Device.class );
        boolean onSameDevice = _shareGuestDevice( inputs );
        boolean doAccel = !_operation.getOperator().equals(",") && onSameDevice;
        return ( doAccel && device != null ) ? device : inputs[ 0 ].getDevice();
    }

    private static boolean _shareGuestDevice( Tsr[] tensors )
    {
        boolean onSameGuestDevice = true;
        Device<?> device = null;
        for ( Tsr<?> tensor : tensors ) device = ( tensor.isOutsourced() ) ? tensor.get( Device.class ) : device;

        if ( device != null ) {
            for ( Tsr<?> tsr : tensors ) {
                onSameGuestDevice = ( !tsr.isVirtual() && device == tsr.get(Device.class) ) && onSameGuestDevice;
            }
        }
        else onSameGuestDevice = false;

        if ( device != null && tensors.length == 2 && tensors[ 1 ].size() == 1 ) onSameGuestDevice = true;
        return onSameGuestDevice;
    }

   //###

    @Override
    public Tsr<?> execute(Tsr<?>... inputs) {
        Args arguments = Args.of(Arg.VarIdx.of(-1), Arg.DerivIdx.of(-1));
        return preprocess(
                    (Tsr<Object>[]) inputs,
                    this,
                    ()-> _tensor_activation( inputs, arguments )
                );
    }

    @Override
    public Tsr<?> execute(Tsr<?>[] inputs, int j) {
        Args arguments = Args.of(Arg.VarIdx.of(j), Arg.DerivIdx.of(-1));
        return preprocess(
                    (Tsr<Object>[]) inputs,
                    this,
                    ()-> _tensor_activation( inputs, arguments )
                );
    }

    @Override
    public Tsr<?> executeDerive(Tsr<?>[] inputs, int d, int j) {
        Args arguments = Args.of(Arg.DerivIdx.of(d), Arg.VarIdx.of(j));
        return preprocess(
                   (Tsr<Object>[]) inputs,
                   this,
                   ()-> _tensor_activation( inputs, arguments )
                );
    }

    @Override
    public Tsr<?> executeDerive(Tsr<?>[] inputs, int d) {
        Args arguments = Args.of( Arg.VarIdx.of(-1), Arg.DerivIdx.of(d) );
        return preprocess(
                    (Tsr<Object>[]) inputs,
                    this,
                    ()-> _tensor_activation( inputs, arguments )
                );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public double call( final double[] inputs, int j ) {
        return this.getOperation().calculate( inputs, j, -1, _src );
    }

    @Override
    public double call( final double... inputs ) {
        return this.getOperation().calculate( inputs, -1, -1, _src );
    }

    @Override
    public double derive( final double[] inputs, final int d, final int j ) {
        return this.getOperation().calculate( inputs, j, d, _src );
    }

    @Override
    public double derive( final double[] inputs, final int d ) {
        return this.getOperation().calculate( inputs, -1, d, _src );
    }

    public Operation getOperation() {
        return this._operation;
    }

    public boolean isFlat() {
        return this._isFlat;
    }

    public boolean isDoingAD() {
        return this._isDoingAD;
    }


    private Tsr<Object> preprocess(
            Tsr<Object>[] inputs,
            Function function,
            Supplier<Tsr<Object>> activation
    ) {
        if ( !function.isDoingAD() ) {
            return activation.get(); // TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES
        }
        boolean allLocked = true; // Input tensors might all have graph nodes which are left from previous computation.
        // ( => needs to be locked again! )
        Tsr<?> untracked = null;
        for ( Tsr<?> t : inputs ) {
            GraphNode<Object> node = t.get( GraphNode.class );
            if ( node != null ) {
                untracked = t;
                allLocked = node.getLock().isLocked() && allLocked;
            }
        }
        if ( untracked == null || !allLocked ) { // If graph tracking (nodes) has not yet been initialized!
            return commit( inputs, function, activation );
        }
        GraphLock lock =  untracked.get( GraphNode.class ).getLock();
        for ( Tsr<Object> t : inputs ) {
            if ( t.has( GraphNode.class ) ) t.get( GraphNode.class ).obtainLocking( lock );
            else new GraphNode( function, lock, () -> t );
        }
        Tsr<Object> result = activation.get();
        return result;
    }

    private static <T> Tsr<T> commit(
            Tsr<?>[] inputs, Function function, Supplier<Tsr<Object>> activation
    ) {
        Tsr.makeFit( inputs, function.isDoingAD() ); // reshaping if needed

        GraphLock newLock = new GraphLock( function );
        for ( Tsr<?> t : inputs ) {
            if ( t.has( GraphNode.class ) ) t.get( GraphNode.class ).obtainLocking( newLock );
            else new GraphNode( function, newLock, () -> t );
        }
        Tsr<T> result;
        if ( activation == null ) result = (Tsr<T>) function.execute( inputs );
        else result = (Tsr<T>) activation.get();

        newLock.release();
        return result;
    }

}