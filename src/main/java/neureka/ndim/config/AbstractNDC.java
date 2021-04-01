package neureka.ndim.config;

import neureka.Neureka;
import neureka.ndim.config.types.complex.*;
import neureka.ndim.config.types.simple.*;
import neureka.ndim.config.types.views.SimpleReshapeView;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;

import java.util.*;

/**
 *  The following is an abstract implementation of the {@link NDConfiguration} which offers a basis for
 *  instantiation and caching of concrete implementations extending this abstract class.
 *  Concrete {@link NDConfiguration} implementations are expected to be immutable which ensures that sharing them is safe.
 *  In order to cash instances based in their field variables, this class comes with a common
 *  implementation of the {@link NDConfiguration#keyCode()} method.
 *  {@link NDConfiguration} implementation instances will be used by tensors which often times
 *  share the same shape, and way of mapping indices to their respective data.
 *  In these cases tensors can simply share their {@link NDConfiguration} instances for memory efficiency.
 */
public abstract class AbstractNDC implements NDConfiguration
{
    /**
     *  Instances implementing the {@link NDConfiguration} interface will be cached in the hashmap below.
     *  In production we can expect a multitude of tensors having the same shape and also the same way of viewing their data.
     *  Therefore they will have configuration instances with the same state.
     *  Implementations of {@link NDConfiguration} are expected to be immutable which allows us to have them be
     *  shared between tensors (because they are read only, meaning no side-effects).
     */
    private static final Map<Long, NDConfiguration> _CACHED_NDCS;
    static
    {
        _CACHED_NDCS = Collections.synchronizedMap( new WeakHashMap<>() ) ;
    }

    /**
     *  The following is a global cache for readonly integer arrays.
     *  Warning! This can of course become dangerous when these arrays are being shared and modified.
     *  Please copy them when exposing them to the user.
     */
    private static final Map<Long, int[]> _CACHED_INT_ARRAYS;
    static
    {
        _CACHED_INT_ARRAYS = Collections.synchronizedMap( new WeakHashMap<>() ) ;
    }

    /**
     *  This method receives an int array and returns an int array which
     *  can either be the one provided or an array found in the global int array cache residing inside
     *  this class.
     *
     * @param data The integer array which ought to be cached.
     * @return The provided array or an already present array found in the int array cache.
     */
    protected static int[] _cacheArray( int[] data )
    {
        long key = 0;
        for ( int e : data ) {
            if ( e <= 10 ) key *= 10;
            else if ( e <= 100 ) key *= 100;
            else if ( e <= 1000 ) key *= 1000;
            else if ( e <= 10000 ) key *= 10000;
            else if ( e <= 100000 ) key *= 100000;
            else if ( e <= 1000000 ) key *= 1000000;
            else if ( e <= 10000000 ) key *= 10000000;
            else if ( e <= 100000000 ) key *= 100000000;
            else if ( e <= 1000000000 ) key *= 1000000000;
            key += Math.abs( e ) + 1;
        }
        int rank = data.length;
        while ( rank != 0 ) {
            rank /= 10;
            key *= 10;
        }
        key += data.length;
        int[] found = _CACHED_INT_ARRAYS.get( key );
        if ( found != null ) return found;
        else {
            _CACHED_INT_ARRAYS.put( key, data );
            return data;
        }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int[] asInlineArray()
    {
        //CONFIG TRANSFER: <[ shape | translation | indicesMap | indices | scale ]>
        int rank = rank();
        int[] inline = new int[ rank * 5 ];
        System.arraycopy( shape(), 0, inline, 0, rank );// -=> SHAPE COPY
        System.arraycopy( translation(), 0, inline, rank * 1, rank );// -=> TRANSLATION COPY
        System.arraycopy( indicesMap(), 0, inline, rank * 2, rank );// -=> IDXMAP COPY (translates scalarization to dimension index)
        System.arraycopy( offset(), 0, inline, rank * 3, rank );// -=> SPREAD
        System.arraycopy( spread(), 0, inline, rank * 4, rank );
        return inline;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public long keyCode()
    {
        return Arrays.hashCode( shape() ) +
               Arrays.hashCode( translation() ) * 2L +
               Arrays.hashCode( indicesMap() ) * 3L +
               Arrays.hashCode( spread() ) * 4L +
               Arrays.hashCode( offset() ) * 5L;
    }

    @Override
    public boolean equals( NDConfiguration ndc )
    {
        return  Arrays.equals(shape(), ndc.shape()) &&
                Arrays.equals(translation(), ndc.translation()) &&
                Arrays.equals(indicesMap(), ndc.indicesMap()) &&
                Arrays.equals(spread(), ndc.spread()) &&
                Arrays.equals(offset(), ndc.offset());
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public static NDConfiguration construct (
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        for ( int dim : shape ) {
            if ( dim == 0 ) {
                String message = "Trying to create tensor configuration containing shape with dimension 0.\n" +
                        "Shape dimensions must be greater than 0!\n";
                throw new IllegalStateException( message );
            }
        }
        if ( Neureka.instance().settings().ndim().isOnlyUsingDefaultNDConfiguration() ) {
            return ComplexDefaultNDConfiguration.construct(shape, translation, indicesMap, spread, offset);
        }
        boolean isSimple = _isSimpleConfiguration(shape, translation, indicesMap, spread, offset);
        NDConfiguration ndc = null;
        if ( isSimple ) {
            if ( shape.length == 1 ) {
                if ( shape[ 0 ]==1 ) ndc = SimpleScalarConfiguration.construct();
                else ndc = SimpleD1Configuration.construct(shape, translation);
            } else if ( shape.length == 2 ) {
                ndc = SimpleD2Configuration.construct(shape, translation);
            } else if ( shape.length == 3 ) {
                ndc = SimpleD3Configuration.construct(shape, translation);
            } else ndc = SimpleDefaultNDConfiguration.construct(shape, translation);
        } else {
            if ( shape.length == 1 ) {
                if ( shape[ 0 ] == 1 ) ndc = ComplexScalarConfiguration.construct(shape, offset);
                else ndc = ComplexD1Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else if ( shape.length == 2 ) {
                ndc = ComplexD2Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else if ( shape.length == 3 ) {
                ndc = ComplexD3Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else ndc = ComplexDefaultNDConfiguration.construct(shape, translation, indicesMap, spread, offset);
        }
        return ndc;
    }

    protected static <T extends NDConfiguration> NDConfiguration _cached( T ndc )
    {
        assert !( ndc instanceof VirtualNDConfiguration );
        long key = ndc.keyCode();
        NDConfiguration found = _CACHED_NDCS.get( key );
        if ( found != null && ndc.equals(found) ) return found;
        else {
            _CACHED_NDCS.put( key, ndc );
            return ndc;
        }
    }

    private static boolean _isSimpleConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        int[] newTranslation = Utility.newTlnOf( shape );
        int[] newSpread = new int[ shape.length ];
        Arrays.fill( newSpread, 1 );
        return  Arrays.equals( translation, newTranslation ) &&
                Arrays.equals( indicesMap, newTranslation ) &&
                Arrays.equals( offset, new int[ shape.length ] ) &&
                Arrays.equals( spread, newSpread );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public String toString() {
        return "(NDConfiguration|@"+Integer.toHexString(hashCode())+"#"+Long.toHexString(keyCode())+"):{ " +
                    "shape : "+Arrays.toString(shape())+", "+
                    "translation : "+Arrays.toString(translation())+", "+
                    "indicesMap : "+Arrays.toString(indicesMap())+", "+
                    "spread : "+Arrays.toString(spread())+", "+
                    "offset : "+Arrays.toString(offset())+" "+
                "}";
    }

    protected static NDConfiguration _simpleReshape( int[] newForm, NDConfiguration ndc )
    {
        int[] newShape = Utility.rearrange( ndc.shape(), newForm );
        int[] newTranslation = Utility.rearrange( ndc.translation(), newShape, newForm );
        int[] newIdxmap = Utility.newTlnOf( newShape );
        int[] newSpread = new int[ newForm.length ];
        for ( int i = 0; i < newForm.length; i++ ) {
            if ( newForm[ i ] < 0 ) newSpread[ i ] = 1;
            else if ( newForm[ i ] >= 0 ) newSpread[ i ] = ndc.spread( newForm[ i ] );
        }
        int[] newOffset = new int[newForm.length];
        for ( int i = 0; i < newForm.length; i++ ) {
            if ( newForm[ i ] < 0 ) newOffset[ i ] = 0;
            else if ( newForm[ i ] >= 0 ) newOffset[ i ] = ndc.offset( newForm[ i ] );
        }
        return AbstractNDC.construct( newShape, newTranslation, newIdxmap, newSpread, newOffset );
    }

    @Override
    public NDConfiguration newReshaped( int[] newForm )
    {
        //TODO : shape check!
        if ( _isSimpleConfiguration( shape(), translation(), indicesMap(), spread(), offset() ) ) {
            return _simpleReshape( newForm, this );
        } else {
            return new SimpleReshapeView( newForm, this );
            //throw new IllegalStateException("Not ready");
        }


    }


}
