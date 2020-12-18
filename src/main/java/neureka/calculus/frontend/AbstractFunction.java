/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*/

package neureka.calculus.frontend;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;
import neureka.calculus.Function;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.implementations.functional.Activation;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.frontend.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.calculus.frontend.implementations.FunctionConstant;
import neureka.calculus.frontend.implementations.FunctionInput;
import neureka.calculus.frontend.implementations.FunctionVariable;

import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Accessors( prefix = {"_"} )
public abstract class AbstractFunction extends AbstractBaseFunction
{
    @Getter
    private final OperationType _operation;
    @Getter
    private final boolean _isFlat;
    @Getter
    private final boolean _isDoingAD;

    private final List<Function> _src;

    //------------------------------------------------------------------------------------------------------------------

    /**
     *
     * @param type
     * @param sources
     * @param doAD
     */
    protected AbstractFunction(OperationType type, List<Function> sources, boolean doAD )
    {
        if( type.getArity() >= 0 && sources.size() != type.getArity() ) {
            String tip = ( type.isIndexer() )
                    ? "\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
                    : "";
            throw new IllegalArgumentException(
                    "The function/operation '"+type.getOperator()+"' expects "+type.getArity()+" parameters, "+
                            "however "+sources.size()+" where given!"+tip
            );
        }
        boolean isFlat = true;
        for ( Function f : sources ) { // AbstractFunction does only reference tip nodes of the function graph:
            isFlat = ((f instanceof FunctionInput) || (f instanceof FunctionVariable) || (f instanceof FunctionConstant)) && isFlat;
        }

        _operation = type;
        _isFlat = isFlat;
        _src = sources;
        _isDoingAD = doAD;
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild( String expression ) {
        return FunctionBuilder.build( expression, true );
    }

    //---

    @Override
    public String toString()
    {
        List<String> stringedSource = _src.stream().map(e->((e==null)?"(null)":e.toString())).collect(Collectors.toList());
        return _operation.getStringifier().asString(stringedSource);
    }

    @Override
    public boolean dependsOn( int index ) {
        for ( Function f : _src ) if ( f.dependsOn(index) ) return true;
        return false;
    }

    //------------------------------------------------------------------------------------------------------------------

    /**
     * Responsible for handling functions with multiple inputs!
     *
     * @param inputs
     * @param j
     * @param d
     * @return
     */
    protected Tsr _tensor_activation( Tsr[] inputs, int j, int d )
    {
        ExecutionCall<Device> call = new ExecutionCall<>(
                _device( inputs ),
                inputs,
                d,
                j,
                _operation
        );
        ExecutionCall<Device> finalCall;
        Device possiblyNewDevice = call.getImplementation().findDeviceFor(call);
        if ( possiblyNewDevice != null ) finalCall = call.withNew(possiblyNewDevice);
        else finalCall = call;

        /* The code below deals with deep functions (non flat) :  */
        if ( _isFlat ) {
            /* The following code is reached in flat functions only:  */
            /* Autograd-Graph will be generated below for the new GraphNode: */
            /* only flat functions can be executed directly */
            if ( d < 0 && _isDoingAD)
                return new GraphNode(this, finalCall, () -> __flat_execution(finalCall)).getPayload();
            else
                return __flat_execution( finalCall );
        }
        else if ( d < 0 ) return __deep_activation( finalCall );
        else return _deep_derivative( finalCall );

    }

    private Tsr __flat_execution( ExecutionCall<Device> call )
    {
        Tsr alternative = call.getImplementation().handleInsteadOfDevice( this, call );
        if ( alternative != null ) return alternative;

        if ( call.getDerivativeIndex() < 0 ) return __deep_activation( call );
        else return _deep_derivative( call  );
    }

    public List<Function> getChildren() {
        return _src;
    }

    private Tsr __deep_activation(ExecutionCall<Device> call )
    {
        Tsr[] inputs = call.getTensors();
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        int j = call.getJ();

        Tsr[] tsrs;
        if ( _operation.isIndexer() ) tsrs = new Tsr[ 1 + inputs.length ];
        else tsrs = new Tsr[ 1 + _src.size() ];

        if ( _operation.isIndexer() ) {
            for ( int i = 1; i < tsrs.length; i++ ) tsrs[ i ] = _src.get( 0 ).call(inputs, i - 1);
        } else if (
                !_isFlat && j < 0 && (
                        _operation.isOperator() || _operation.supportsImplementation(Activation.class)
                )
        ) {/*   '+', '-', 'x', '*', '%', '«', '»', ',', ...   */
            tsrs = srcActivation(inputs, j, d, 0);
            List<String> stringedSource = IntStream.range(0, _src.size()).mapToObj(i -> "I[" + i + "]").collect(Collectors.toList());
            String asStr = _operation.getStringifier().asString(stringedSource);
            return FunctionBuilder.build(asStr, _isDoingAD).call(tsrs);
        } else {
            tsrs = srcActivation(inputs, j, d, 1);
        }
        device.execute( new ExecutionCall<>( device, tsrs, d, _operation) );

        return ( tsrs[ 0 ] == null ) ? tsrs[ 1 ] : tsrs[ 0 ];
    }

    /**
     *  This method return the index of the tensor
     *  in the given tensor array which is virtual and contains "1.0".
     *  However if not all tensors are virtual or their values are not all "0.0" except one
     *  whose value is "1.0" then it return -1, because the optimization cannot
     *  be made...
     *
     * @param tsrs An array of tensors which ought to be analyzed.
     * @return The index of the tensor whose value is "1.0" (if all other are "0.0"), otherwise : -1
     */
    private int ___indexOfFoundDerivative( Tsr[] tsrs )
    {
        boolean allVirtual = true;
        for ( Tsr t : tsrs ) if ( t != null && !t.isVirtual() ) allVirtual = false;
        if ( allVirtual ) {
            int index = -1;
            for ( int i=0; i < tsrs.length; i++ ) {
                double value = ( tsrs[ i ] == null ) ? 0.0 : tsrs[ i ].value64( 0 );
                if ( value == 1.0 ) {
                    if ( index >= 0 ) return -1;
                    index = i;
                } else if ( value != 0.0 ) return -1;
            }
            return index;
        }
        return -1;
    }

    private Tsr _deep_derivative( ExecutionCall<Device> call )
    {
        Supplier<Tsr> actor =
        () ->
        {
            Tsr[] inputs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            int j = call.getJ();

            Tsr[] tsrs;
            if ( _operation.isIndexer() ) tsrs = new Tsr[ 1 + inputs.length ];
            else tsrs = new Tsr[ 1 + _src.size() ];

            // Chain-rule (forward AutoDiff):
            // inner times outer means:
            // first derive source!
            // like so:
            if ( _operation.isIndexer() ) {
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[ i ] = _src.get( 0 ).derive(inputs, d, i - 1);
                }
            } else {
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[ i ] = ( j >= 0 ) ? _src.get( i - 1 ).derive( inputs, d, j ) : _src.get( i - 1 ).derive( inputs, d );
                }
            }
            //...then add them all together! (is possible because of linearity...)
            Tsr inner;
            if ( tsrs.length > 2 ) {// Optimization: Finds index of "1.0" among otherwise all "0.0" virtual tensors!
                int index = ___indexOfFoundDerivative( tsrs );
                if ( index >= 0 ) inner = tsrs[index];
                else {
                    // Optimization above did not apply, so we accumulate all the derivatives!
                    device.execute( new ExecutionCall<>( device, tsrs, -1, OperationType.instance("+") ) );
                    inner = tsrs[ 0 ];//this is now the inner derivative!
                }
            } else inner = tsrs[ 1 ];

            tsrs[ 0 ] = null;
            //...then activate (No differentiation!) the source like so:
            if ( _operation.isIndexer() ) { // Indexer pass an index j of course!
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[ i ] = _src.get( 0 ).call( inputs, i - 1 ); // i - 1 := j
                }
            } else {
                for ( int i = 1; i < tsrs.length; i++ ) {
                    tsrs[ i ] = ( j >= 0 ) ? _src.get(i - 1).call( inputs, j ) : _src.get(i - 1).call( inputs );
                }
            }
            //...get derivative index within src list:
            for ( int i = 0; i < _src.size(); i++ ) {
                if ( _src.get( i ).dependsOn(d) && !_operation.isIndexer() ) {
                    d = i;
                    break;
                }
            }
            // Use those tensors for the outer derivative:
            device.execute( new ExecutionCall<>( device, tsrs, d, _operation) );
            // At the end:
            //...multiply inner times outer: ( if inner is not 1 entirely... )
            if ( !( ( inner.isVirtual() || inner.size()==1 ) && inner.value64( 0 )==1.0) ) {
                tsrs = new Tsr[]{null, inner, tsrs[ 0 ]};
                device.execute( new ExecutionCall<>( device, tsrs, -1, OperationType.instance("*") ) );
            } // done!
            return tsrs[ 0 ];

        };
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        Tsr out = null;
        for ( int i = 0; i < _src.size(); i++ ) { // constants need to be figured out!
            int di = ( _src.get( i ).dependsOn(d) ) ? i : -1;
            if ( di >= 0 ) {
                if ( out == null ) out = actor.get();
                else device.execute(
                        new ExecutionCall<>(
                                device, new Tsr[]{null, actor.get(), out}, -1, OperationType.instance("+")
                        )
                );
            }
        }
        return out;
    }

    public Tsr[] srcActivation( Tsr[] inputs, int j, int d, int offset )
    {
        int[] tempShape = null;
        Tsr[] tsrs = new Tsr[ _src.size() + offset ];
        for ( int i = offset; i < tsrs.length; i++ ) {//constants need to be figured out!
            if ( !(_src.get(i - offset) instanceof FunctionConstant) ) {
                if ( d < 0 ) {
                    tsrs[ i ] = ( j >= 0 ) ? _src.get(i - offset).call( inputs, j ) : _src.get(i - offset).call( inputs );
                } else {
                    tsrs[ i ] = ( j >= 0 ) ? _src.get(i - offset).derive( inputs, d, j ) : _src.get(i - offset).derive( inputs, d );
                }
                tempShape = ( tempShape == null ) ? tsrs[ i ].getNDConf().shape() : tempShape;
            }
        }
        for ( int i = offset; i < tsrs.length; i++ ) {
            if ( tsrs[ i ] == null ) {
                tsrs[ i ] =
                        ( j < 0 )
                                ? new Tsr(tempShape, ((FunctionConstant) _src.get(i - offset)).value())
                                : new Tsr(tempShape, _src.get(i - offset).call(new double[]{}, j));
            }
        }
        return tsrs;
    }

    private Device _device( Tsr<Object>[] inputs )
    {
        if ( inputs.length == 0 ) return HostCPU.instance();
        Device device = inputs[ 0 ].find( Device.class );
        boolean onSameDevice = _shareGuestDevice( inputs );
        boolean doAccel = !_operation.getOperator().equals(",") && onSameDevice;
        return ( doAccel && device != null ) ? device : inputs[ 0 ].device();
    }

    private static boolean _shareGuestDevice( Tsr[] tsrs )
    {
        boolean onSameGuestDevice = true;
        Device device = null;
        for ( Tsr<Object> tsr : tsrs ) device = ( tsr.isOutsourced() ) ? tsr.find( Device.class ) : device;

        if ( device != null ) {
            for ( Tsr tsr : tsrs ) {
                onSameGuestDevice = ( !tsr.isVirtual() && device == tsr.find(Device.class) ) && onSameGuestDevice;
            }
        } else onSameGuestDevice = false;

        if ( device != null && tsrs.length == 2 && tsrs[ 1 ].size() == 1 ) {
            onSameGuestDevice = true;
        }
        return onSameGuestDevice;
    }

}
