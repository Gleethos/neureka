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

import neureka.Data;
import neureka.Tensor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.common.composition.Component;
import neureka.common.utility.DataConverter;
import neureka.dtype.DataType;
import neureka.framing.Relation;
import neureka.math.args.Arg;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  This is the abstract precursor class providing
 *  some useful implementations for core concepts which are most likely
 *  applicable to most concrete implementations of the {@link Device} interface.
 *
 * @param <V> The common super type for the types of tensors storable on this device.
 */
public abstract class AbstractDevice<V> extends AbstractBaseDevice<V>
{
    private static final DeviceCleaner _CLEANER = DeviceCleaner.INSTANCE;

    protected Logger _log;


    protected AbstractDevice() { _log = LoggerFactory.getLogger( getClass() ); }

    /**
     *  This method is the internal approval routine called by its public counterpart
     *  and implemented by classes extending this very abstract class.
     *  It may or may not be called by an {@link Algorithm}
     *  in order to allow a {@link Device} to checked if the provided arguments are suitable for execution.
     *
     * @param tensors An array of input tensors.
     * @param d The index of the input which ought to be derived.
     * @param type The type of operation.
     * @return The truth value determining if the provided arguments can be executed.
     */
    protected abstract boolean _approveExecutionOf(Tensor<?>[] tensors, int d, Operation type );

    /**
     *  A {@link Device} is a component of a tensor. This method is used to inform the device
     *  that the device is being added, removed or replaced (from the tensor).
     *
     * @param changeRequest An {@link OwnerChangeRequest} implementation instance used to communicate the type of change, context information and the ability to execute the change directly.
     * @return The truth value determining if the change should be executed.
     */
    @Override
    public boolean update( OwnerChangeRequest<Tensor<V>> changeRequest ) {
        Tensor<V> oldOwner = changeRequest.getOldOwner();
        Tensor<V> newOwner = changeRequest.getNewOwner();
        if ( changeRequest.type() == IsBeing.REPLACED ) _swap( oldOwner, newOwner );
        else if ( changeRequest.type() == IsBeing.ADDED ) {
            if ( newOwner.has( Relation.class ) ) {
                Relation<V> relation = newOwner.get(Relation.class);
                if ( relation.hasParent() ) { // Root needs to be found ! :
                    Tensor<V> root = relation.findRootTensor().orElseThrow(IllegalStateException::new);
                    if ( !this.has(root) || !root.isOutsourced() )
                        throw new IllegalStateException("Data parent is not outsourced!");
                }
            }

            Device<V> found = newOwner.getMut().getData().owner();

            if ( found != null && found != this )
                found.restore( newOwner );
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
        if ( !_approveExecutionOf( call.inputs(), call.getValOf( Arg.DerivIdx.class ), call.getOperation() ) )
            throw new IllegalArgumentException("Provided execution call has not been approved by this device.");

        return this;
    }

    /** {@inheritDoc} */
    @Override
    public <T extends V> Storage<V> store( Tensor<T> tensor ) {
        tensor.set( (Component) this ); // This way we move the storing procedure to the update function!
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public <T extends V> Access<T> access( Tensor<T> tensor ) {
        return new Access<T>() {
            @Override public Writer write( T item ) {
                return new Writer() {
                    @Override public void intoRange( int start, int limit ) { _writeItemInternal( tensor, item, start, limit-start ); }
                    @Override public void fully() { _writeItemInternal( tensor, item, 0, tensor.size() ); }
                };
            }
            @Override public Writer writeFrom( Object array, int offset ) {
                return new Writer() {
                    @Override public void intoRange( int start, int limit ) { _writeArrayInternal( tensor, array, offset, start, limit-start ); }
                    @Override public void fully() { _writeArrayInternal( tensor, array, offset, 0, tensor.size() ); }
                };
            }
            @Override public T readAt( int index ) { return _readItem( tensor, index ); }
            @Override public <A> A readArray( Class<A> arrayType, int start, int size ) { return _readArray( tensor, arrayType, start, size ); }
            @Override public Object readAll( boolean clone ) { return _readAll( tensor, clone ); }
            @Override public int getDataSize() { return _sizeOccupiedBy( tensor ); }
            @Override public void cleanup( Runnable action ) { _cleaning( tensor, action ); }
            @Override public Data<V> actualize() { return _actualize( tensor ); }
            @Override public Data<V> virtualize() { return _virtualize( tensor ); }
        };
    }

    private  <T extends V> void _writeItemInternal(Tensor<T> tensor, T item, int start, int size ) {
        Class<T> itemType = tensor.itemType();
        if ( !itemType.isAssignableFrom( item.getClass() ) )
            item = DataConverter.get().convert( item, itemType );
        _writeItem( tensor, item, start, size );
    }

    private <T extends V> void _writeArrayInternal(
            Tensor<T> tensor, Object array,
            int offset, int start, int size
    ) {
        DataType<?> dataType = tensor.getDataType();
        if ( dataType == null )
            dataType = _dataTypeOf( array );
        Class<?> arrayType = dataType.dataArrayType();
        if ( !arrayType.isAssignableFrom( array.getClass() ) )
            array = DataConverter.get().convert( array, arrayType );
        _writeArray( tensor, array, offset, start, size );
    }

    /**
     *  This method is used internally mostly and should not be used in most cases.    <br><br>
     *
     * @param <T> The type parameter for the value type of the tensors, which must be supported by this {@link Device}.
     * @param former The tensor whose associated data (on the device) ought to be assigned to the other tensor.
     * @param replacement The tensor which ought to receive the data of the former tensor internally.
     */
    protected abstract <T extends V> void _swap(Tensor<T> former, Tensor<T> replacement );

    protected abstract <T extends V> int _sizeOccupiedBy( Tensor<T> tensor );

    protected abstract <T extends V> Object _readAll(Tensor<T> tensor, boolean clone );

    protected abstract <T extends V> T _readItem(Tensor<T> tensor, int index );

    protected abstract <T extends V, A> A _readArray(Tensor<T> tensor, Class<A> arrayType, int start, int size );

    protected abstract <T extends V> void _writeItem(Tensor<T> tensor, T item, int start, int size );

    protected abstract <T extends V> void _writeArray(Tensor<T> tensor, Object array, int offset, int start, int size );

    protected abstract Data<V> _actualize( Tensor<?> tensor );

    protected abstract Data<V> _virtualize( Tensor<?> tensor );

    protected abstract DataType<?> _dataTypeOf( Object rawData );

}
