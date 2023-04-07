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

           _         _                  _   ____                 _____             _
     /\   | |       | |                | | |  _ \               |  __ \           (_)
    /  \  | |__  ___| |_ _ __ __ _  ___| |_| |_) | __ _ ___  ___| |  | | _____   ___  ___ ___
   / /\ \ | '_ \/ __| __| '__/ _` |/ __| __|  _ < / _` / __|/ _ \ |  | |/ _ \ \ / / |/ __/ _ \
  / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |_) | (_| \__ \  __/ |__| |  __/\ V /| | (_|  __/
 /_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|____/ \__,_|___/\___|_____/ \___| \_/ |_|\___\___|


*/

package neureka.devices;

import neureka.Data;
import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.dtype.DataType;
import neureka.ndim.config.NDConfiguration;

import java.util.Collection;
import java.util.Iterator;
import java.util.Spliterator;

/**
 * @param <V> The value type parameter representing a common super type for all values supported by the device.
 */
public abstract class AbstractBaseDevice<V> implements Device<V>
{
    @Override
    public int size() {
        Collection<Tsr<V>> tensors = this.getTensors();
        if ( tensors == null ) return 0;
        return tensors.size();
    }

    /**
     *  A device is empty if there are no tensors stored on it.
     *
     * @return The truth value determining if there are no tensors stored on this device.
     */
    @Override
    public boolean isEmpty() { return this.size() == 0; }

    @Override
    public boolean contains( Tsr<V> o ) { return this.getTensors().contains( o ); }

    @Override
    public Iterator<Tsr<V>> iterator() { return this.getTensors().iterator(); }

    @Override
    public Spliterator<Tsr<V>> spliterator() { return getTensors().spliterator(); }

    protected abstract static class DeviceData<T> implements Data<T>
    {
        protected final Device<?> _owner;
        protected final Object _dataRef;
        protected final DataType<T> _dataType;

        public DeviceData(
                Device<?> owner,
                Object ref,
                DataType<T> dataType
        ) {
            LogUtil.nullArgCheck( owner, "owner", Device.class );
            LogUtil.nullArgCheck( dataType, "dataType", DataType.class );
            _owner = owner;
            _dataRef = ref;
            _dataType = dataType;
        }

        @Override public final Device<T> owner() { return (Device<T>) _owner; }
        @Override public final Object getRef() { return _dataRef; }
        @Override public final DataType<T> dataType() { return _dataType; }
        @Override public       Data<T> withNDConf(NDConfiguration ndc) { return this; }
    }

}
