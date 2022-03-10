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

   _   _ _____ _____ _                 _
  | \ | |  __ \_   _| |               | |
  |  \| | |  | || | | |_ ___ _ __ __ _| |_ ___  _ __
  | . ` | |  | || | | __/ _ \ '__/ _` | __/ _ \| '__|
  | |\  | |__| || |_| ||  __/ | | (_| | || (_) | |
  |_| \_|_____/_____|\__\___|_|  \__,_|\__\___/|_|


*/


package neureka.ndim.iterator;

import neureka.Tsr;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.reshaped.Reshaped2DConfiguration;
import neureka.ndim.config.types.reshaped.Reshaped3DConfiguration;
import neureka.ndim.config.types.simple.Simple1DConfiguration;
import neureka.ndim.config.types.simple.Simple2DConfiguration;
import neureka.ndim.config.types.simple.Simple3DConfiguration;
import neureka.ndim.config.types.sliced.Sliced1DConfiguration;
import neureka.ndim.config.types.sliced.Sliced2DConfiguration;
import neureka.ndim.config.types.sliced.Sliced3DConfiguration;
import neureka.ndim.config.types.views.virtual.VirtualNDConfiguration;
import neureka.ndim.iterator.types.reshaped.Reshaped2DCIterator;
import neureka.ndim.iterator.types.reshaped.Reshaped3DCIterator;
import neureka.ndim.iterator.types.simple.Simple1DCIterator;
import neureka.ndim.iterator.types.simple.Simple2DCIterator;
import neureka.ndim.iterator.types.simple.Simple3DCIterator;
import neureka.ndim.iterator.types.sliced.Sliced1DCIterator;
import neureka.ndim.iterator.types.sliced.Sliced2DCIterator;
import neureka.ndim.iterator.types.sliced.Sliced3DCIterator;
import neureka.ndim.iterator.types.sliced.SlicedNDIterator;
import neureka.ndim.iterator.types.virtual.VirtualNDIterator;

/**
 *  An {@link NDIterator} is used to iterate over n-dimensional arrays.
 *  Their implementations are based on specific {@link NDConfiguration}
 *  implementations which define the access pattern for a nd-array / tensor.
 *  This functionality is abstracted away by these 2 interfaces in order
 *  to allow for specialize implementations for various types
 *  of access patterns for various types of dimensionality...
 */
public interface NDIterator
{
    /**
     *  Defines if a new {@link NDIterator} is allowed to be a {@link VirtualNDIterator}.
     */
    enum NonVirtual { TRUE, FALSE }

    /**
     *  Use this to instantiate {@link NDIterator}s optimized for the provided tensor.
     *
     * @param t The tensor for which an optimized {@link NDIterator} should be created.
     * @return A new {@link NDIterator} instance optimized for the provided tensor.
     */
    static NDIterator of( Tsr<?> t ) {
        return of( t, NonVirtual.FALSE );
    }

    /**
     *  Use this to instantiate {@link NDIterator}s optimized for the provided tensor
     *  which may not be allowed to be a {@link VirtualNDIterator} instance.
     *
     * @param t The tensor for which an optimized {@link NDIterator} should be created.
     * @param shouldNotBeVirtual The enum which determines if a virtual iterator is allowed.
     * @return A new {@link NDIterator} instance optimized for the provided tensor.
     */
    static NDIterator of( Tsr<?> t, NonVirtual shouldNotBeVirtual ) {
        return of( t.getNDConf(), shouldNotBeVirtual );
    }

    /**
     *  Use this to instantiate {@link NDIterator}s optimized for the provided {@link NDConfiguration}
     *  which may not be allowed to be a {@link VirtualNDIterator} instance.
     *
     * @param ndc The nd-config for which an optimized {@link NDIterator} should be created.
     * @param shouldNotBeVirtual The enum which determines if a virtual iterator is allowed.
     * @return A new {@link NDIterator} instance optimized for the provided {@link NDConfiguration}.
     */
    static NDIterator of( NDConfiguration ndc, NonVirtual shouldNotBeVirtual ) {

        if ( ndc instanceof Simple1DConfiguration   ) return new Simple1DCIterator(     (Simple1DConfiguration) ndc );
        if ( ndc instanceof Sliced1DConfiguration   ) return new Sliced1DCIterator(     (Sliced1DConfiguration) ndc );

        if ( ndc instanceof Simple2DConfiguration   ) return new Simple2DCIterator(     (Simple2DConfiguration) ndc );
        if ( ndc instanceof Reshaped2DConfiguration ) return new Reshaped2DCIterator( (Reshaped2DConfiguration) ndc );
        if ( ndc instanceof Sliced2DConfiguration   ) return new Sliced2DCIterator(     (Sliced2DConfiguration) ndc );

        if ( ndc instanceof Simple3DConfiguration   ) return new Simple3DCIterator(      (Simple3DConfiguration) ndc );
        if ( ndc instanceof Reshaped3DConfiguration ) return new Reshaped3DCIterator(  (Reshaped3DConfiguration) ndc );
        if ( ndc instanceof Sliced3DConfiguration   ) return new Sliced3DCIterator(      (Sliced3DConfiguration) ndc );

        if ( ndc instanceof VirtualNDConfiguration && shouldNotBeVirtual == NonVirtual.FALSE )
            return new VirtualNDIterator( (VirtualNDConfiguration) ndc );
        else
            return new SlicedNDIterator( ndc );
    }


    int shape( int i );

    int[] shape();

    void increment();

    default int getIndexAndIncrement() {
        int i = i();
        this.increment();
        return i;
    }

    void decrement();

    int i();

    int get( int axis );

    int[] get();

    void set( int axis, int position );

    void set( int[] indices );

    int rank();

}
