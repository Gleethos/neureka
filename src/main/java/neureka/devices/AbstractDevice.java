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

            _         _                  _   _____             _
      /\   | |       | |                | | |  __ \           (_)
     /  \  | |__  ___| |_ _ __ __ _  ___| |_| |  | | _____   ___  ___ ___
    / /\ \ | '_ \/ __| __| '__/ _` |/ __| __| |  | |/ _ \ \ / / |/ __/ _ \
   / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |__| |  __/\ V /| | (_|  __/
  /_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|_____/ \___| \_/ |_|\___\___|


*/

package neureka.devices;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.args.Arg;
import neureka.common.composition.Component;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import neureka.framing.Relation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 *  This is the abstract precursor class providing
 *  some useful implementations for core concepts which are most likely
 *  applicable to most concrete implementations of the Device interface.
 *  These class provides the following features :
 *
 *  - A Cleaner instance used for freeing resources of the device.
 *
 *  - An component update implementations which simply calls the swap method of the device.
 *
 *  - An implementation for the execution method which calls the underlying calculus backend.
 *
 * @param <V> The most common super type for all tensors storable on this device.
 */
public abstract class AbstractDevice<V> extends AbstractBaseDevice<V>
{
    private static final DeviceCleaner _CLEANER = new CustomDeviceCleaner();

    protected Logger _log;


    protected AbstractDevice() { _log = LoggerFactory.getLogger( getClass() ); }

    /**
     *  This method is the internal approval routine called by it's public counterpart
     *  and implemented by classes extending this very abstract class.
     *  It may or may not be called by an {@link Algorithm}
     *  in order to allow a {@link Device} to checked if the provided arguments are suitable for execution.
     *
     * @param tensors An array of input tensors.
     * @param d The index of the input which ought to be derived.
     * @param type The type of operation.
     * @return The truth value determining if the provided arguments can be executed.
     */
    protected abstract boolean _approveExecutionOf( Tsr<?>[] tensors, int d, Operation type );


    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        Tsr<V> oldOwner = changeRequest.getOldOwner();
        Tsr<V> newOwner = changeRequest.getNewOwner();
        if ( changeRequest.type() == IsBeing.REPLACED ) _swap( oldOwner, newOwner );
        else if ( changeRequest.type() == IsBeing.ADDED ) {
            if ( newOwner.has( Relation.class ) ) {
                Relation<V> relation = newOwner.get(Relation.class);
                if ( relation.hasParent() ) { // Root needs to be found ! :
                    Tsr<V> root = relation.findRootTensor();
                    if (!this.has(root) || !root.isOutsourced())
                        throw new IllegalStateException("Data parent is not outsourced!");
                }
            }
            Device<V> found = newOwner.getDevice();
            if ( found != null && found != this ) {
                found.restore( newOwner );
            }
        }
        return true;
    }

    protected void _cleaning( Object o, Runnable action ) { _CLEANER.register( o, action ); }

    /**
     *  <b>This method plays an important role in approving a provided {@link ExecutionCall}.</b>
     *  When implementing custom operations or such for the backend of this library, then one may use
     *  this in order to check if the provided call is suitable for this {@link Device}.
     *
     * @param call The execution call object containing tensor arguments and settings for the device to approve.
     * @return This very device instance in order to enable method chaining.
     */
    @Override
    public Device<V> approve( ExecutionCall<? extends Device<?>> call )
    {
        if ( !_approveExecutionOf( call.inputs(), call.getValOf( Arg.DerivIdx.class ), call.getOperation() ) ) {
            throw new IllegalArgumentException("Provided execution call has not been approved by this device.");
        }
        return this;
    }

    @Override
    public <T extends V> Storage<V> store( Tsr<T> tensor ) {
        tensor.set( (Component) this ); // This way we move the storing procedure to the update function!
        return this;
    }

    @Override
    public <T extends V> Access<T> access( Tsr<T> tensor ) {
        return new Access<T>() {
            @Override public Writer write(T item) {
                return new Writer() {
                    @Override public void intoRange(int start, int limit) { _writeItem( tensor, item, start, limit-start ); }
                    @Override public void fully() { _writeItem( tensor, item, 0, tensor.size() ); }
                };
            }
            @Override public Writer writeFrom( Object array, int offset ) {
                return new Writer() {
                    @Override public void intoRange( int start, int limit ) { _writeArray( tensor, array, offset, start, limit-start ); }
                    @Override public void fully() { _writeArray( tensor, array, offset, 0, tensor.size() ); }
                };
            }
            @Override public T readAt( int index ) { return _readItem( tensor, index ); }
            @Override public <A> A readArray( Class<A> arrayType, int start, int size ) { return _readArray( tensor, arrayType, start, size ); }
            @Override public Object readAll( boolean clone ) { return _readAll( tensor, clone ); }
            @Override public int getDataSize() { return _sizeOccupiedBy( tensor ); }
            @Override public void cleanup( Runnable action ) { _cleaning( tensor, action ); }
            @Override public void updateNDConf() { _updateNDConf( tensor ); }
            @Override public Object allocate( int size ) { return _allocate( tensor.getDataType(), size ); }
            @Override public Object actualize() { return _actualize( tensor ); }
        };
    }

    /**
     *  This method is used internally mostly and should not be used in most cases.    <br><br>
     *
     * @param <T> The type parameter for the value type of the tensors, which must be supported by this {@link Device}.
     * @param former The tensor whose associated data (on the device) ought to be assigned to the other tensor.
     * @param replacement The tensor which ought to receive the data of the former tensor internally.
     */
    protected abstract <T extends V> void _swap( Tsr<T> former, Tsr<T> replacement );

    protected abstract <T extends V> void _updateNDConf( Tsr<T> tensor );

    protected abstract <T extends V> int _sizeOccupiedBy( Tsr<T> tensor );

    protected abstract <T extends V> Object _readAll( Tsr<T> tensor, boolean clone );

    protected abstract <T extends V> T _readItem( Tsr<T> tensor, int index );

    protected abstract <T extends V, A> A _readArray( Tsr<T> tensor, Class<A> arrayType, int start, int size );

    protected abstract <T extends V> void _writeItem( Tsr<T> tensor, T item, int start, int size );

    protected abstract <T extends V> void _writeArray( Tsr<T> tensor, Object array, int offset, int start, int size );

    protected abstract Object _allocate( DataType<?> dataType, int size );

    protected Object _actualize( Tsr<?> tensor ) {
        Object value = tensor.getUnsafe().getData();
        DataType<?> dataType = tensor.getDataType();
        int size = tensor.size();
        Class<?> typeClass = dataType.getRepresentativeType();
        Object newValue;
        if ( typeClass == F64.class ) {
            if ( ( (double[]) value ).length == size ) return value;
            newValue = new double[ size ];
            if ( ( (double[]) value )[ 0 ] != 0d ) Arrays.fill( (double[]) newValue, ( (double[]) value )[ 0 ] );
        } else if ( typeClass == F32.class ) {
            if ( ( (float[]) value ).length == size ) return value;
            newValue = new float[size];
            if ( ( (float[]) value )[ 0 ] != 0f ) Arrays.fill( (float[]) newValue, ( (float[]) value )[ 0 ] );
        } else if ( typeClass == I32.class ) {
            if ( ( (int[]) value ).length == size ) return value;
            newValue = new int[ size ];
            if ( ( (int[]) value )[ 0 ] != 0 ) Arrays.fill( (int[]) newValue, ( (int[]) value )[ 0 ] );
        } else if ( typeClass == I16.class ) {
            if ( ( (short[]) value ).length == size ) return value;
            newValue = new short[ size ];
            if ( ( (short[]) value )[ 0 ] != 0 ) Arrays.fill( (short[]) newValue, ( (short[]) value )[ 0 ] );
        } else if ( typeClass == I8.class ) {
            if ( ( (byte[]) value ).length == size ) return value;
            newValue = new byte[ size ];
            if ( ( (byte[]) value )[ 0 ] != 0 ) Arrays.fill( (byte[]) newValue, ( (byte[]) value )[ 0 ] );
        } else if ( typeClass == I64.class ) {
            if ( ( (long[]) value ).length == size ) return value;
            newValue = new long[ size ];
            if ( ( (long[]) value )[ 0 ] != 0 ) Arrays.fill( (long[]) newValue, ( (long[]) value )[ 0 ] );
        } else if ( typeClass == Boolean.class ) {
            if ( ( (boolean[]) value ).length == size ) return value;
            newValue = new boolean[ size ];
            Arrays.fill( (boolean[]) newValue, ( (boolean[]) value )[ 0 ] );
        } else if ( typeClass == Character.class ) {
            if ( ( (char[]) value ).length == size ) return value;
            newValue = new char[ size ];
            if ( ( (char[]) value )[ 0 ] != (char) 0 ) Arrays.fill( (char[]) newValue, ( (char[]) value )[ 0 ] );
        } else {
            if ( ( (Object[]) value ).length == size ) return value;
            newValue = new Object[ size ];
            if ( ( (Object[]) value )[ 0 ] != null ) Arrays.fill( (Object[]) newValue, ( (Object[]) value )[ 0 ] );
        }
        return newValue;
    }
}
