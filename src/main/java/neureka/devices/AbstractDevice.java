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
import neureka.framing.Relation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  This is the abstract precursor class providing
 *  some useful implementations for core concepts which are most likely
 *  applicable to most concrete implementations of the Device interface.
 *  These class provides the following features :
 *
 *  - A Cleaner instance used for freeing resources of the device.
 *
 *  - A component update implementation which simply calls the swap method of the device.
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
            Device<V> found = newOwner.getUnsafe().getDataArray().owner();
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


    /**
     * This method checks if the passed tensor
     * is stored on this {@link Device} instance.
     * "Stored" means that the data of the tensor was created by this device.
     * This data is referenced inside the tensor...
     *
     * @param tensor The tensor in question.
     * @return The truth value of the fact that the provided tensor is on this device.
     */
    @Override
    public <T extends V> boolean has(Tsr<T> tensor) {
        return tensor.getUnsafe().getDataArray() != null && tensor.getUnsafe().getDataArray().owner() == this;
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
            @Override public neureka.Data<V> actualize() { return _actualize( tensor ); }
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

    protected abstract neureka.Data<V> _actualize(Tsr<?> tensor );

    protected neureka.Data<V> _dataArrayOf(Object data ) {
        assert !(data instanceof neureka.Data);
        return new neureka.Data<V>() {
            @Override public Device<V> owner() { return AbstractDevice.this; }
            @Override public Object get() { return data; }
        };
    }

}
